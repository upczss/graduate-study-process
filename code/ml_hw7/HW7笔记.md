# HW7：使用 BERT 完成中文问答

这次作业是中文抽取式问答。模型会接收一个问题和一段文章，然后从文章中找出
答案的开始与结束位置。整体流程是：处理数据、使用 BERT 训练、筛选答案，
最后生成 Kaggle 需要的 `result.csv`。

## 一、读取和分词

```python
with open("hw7_train.json", "r", encoding="utf-8") as file:
    data = json.load(file)

questions = data["questions"]
paragraphs = data["paragraphs"]
```

数据分成问题和文章两部分。每个问题使用 `paragraph_id` 指向对应文章，训练集
还提供 `answer_start`、`answer_end` 和正确答案。

```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese",
    use_fast=True,
)
```

这里使用中文 BERT 的 Fast Tokenizer。除了把文字转换成 token，它还能记录每个
token 在原文中的字符位置，方便最后从原文截取答案。

```python
encoded = tokenizer(
    question_text,
    paragraph,
    truncation="only_second",
    max_length=384,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```

BERT 无法一次读取无限长的文章，所以这里只截断文章，不截断问题。文章过长时，
Tokenizer 会自动切成多个有重叠的窗口。

## 二、滑动窗口

```python
max_length = 384
doc_stride = 128
```

`max_length` 是一次输入 BERT 的最大长度。`doc_stride` 控制窗口之间的重叠，
可以减少答案刚好位于窗口边界时被漏掉的情况。

```python
sequence_ids = encoded.sequence_ids(window_index)

context_indices = [
    index
    for index, sequence_id in enumerate(sequence_ids)
    if sequence_id == 1
]
```

一个输入中同时包含问题和文章。`sequence_id == 1` 表示这个 token 来自文章。
只有文章中的 token 可以成为答案，问题、`[CLS]` 和 `[SEP]` 不能成为答案。

```python
answer_start = question["answer_start"]
answer_end = question["answer_end"] + 1

if answer_start < window_start or answer_end > window_end:
    continue
```

训练时已经知道正确答案的位置。这里只保留包含完整答案的窗口，再把答案的字符
位置转换为窗口中的 token 位置。

## 三、BERT 问答模型

```python
model = AutoModelForQuestionAnswering.from_pretrained(
    "bert-base-chinese"
)
```

这里不是从头训练 BERT，而是在预训练中文 BERT 的基础上进行微调。问答模型会为
每个 token 输出一个开始分数和一个结束分数。

```python
output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch["token_type_ids"],
    start_positions=batch["start_positions"],
    end_positions=batch["end_positions"],
)
```

`input_ids` 是 token 编号，`attention_mask` 用来忽略 PAD，`token_type_ids`
区分问题和文章。训练时再提供正确的开始与结束位置，模型会自动计算 loss。

```python
start_logits = output.start_logits
end_logits = output.end_logits
loss = output.loss
```

`start_logits` 表示每个位置作为答案开头的分数，`end_logits` 表示作为答案结尾
的分数。训练目标就是让正确位置的分数变高。

## 四、训练部分

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=0.01,
)
```

训练使用 AdamW 更新模型。BERT 微调通常使用较小的学习率，学习率过大可能破坏
预训练模型原来学到的参数。

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
```

训练开始时先让学习率慢慢升高，这一步叫 warmup；之后再线性下降到接近 0，
让后期训练更加稳定。

```python
with torch.cuda.amp.autocast(enabled=use_fp16):
    output = model(**batch)
    loss = output.loss / gradient_accumulation

scaler.scale(loss).backward()
```

有 GPU 时可以使用 FP16，减少显存并提高训练速度。把 loss 除以累积次数，是为了
配合 Gradient Accumulation。

```python
if step % gradient_accumulation == 0:
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
```

如果显存放不下较大的 batch，可以连续累积几个小 batch 的梯度，再统一更新
一次模型，效果接近使用更大的 batch size。

## 五、答案后处理

```python
start_candidates = start_logits.argsort()[-20:][::-1]
end_candidates = end_logits.argsort()[-20:][::-1]
```

这里没有直接分别选择分数最大的开始和结束位置，而是各取前 20 个候选，再比较
不同组合。这样更容易找到整体分数较高且合理的答案。

```python
if (
    offsets[start_index] is None
    or offsets[end_index] is None
    or end_index < start_index
    or end_index - start_index + 1 > 40
):
    continue
```

后处理会排除不合理答案：答案必须来自文章，结束位置不能早于开始位置，而且不能
超过最大长度。这解决了原始示例可能产生反向答案的问题。

```python
score = start_logits[start_index] + end_logits[end_index]

if score > best_score:
    best_score = score
    best_answer = paragraph[char_start:char_end]
```

同一个问题可能被切成多个窗口。程序比较所有窗口中的合法答案，并选择开始分数
与结束分数之和最高的结果。

```python
char_start = offsets[start_index][0]
char_end = offsets[end_index][1]
answer = paragraph[char_start:char_end]
```

答案根据字符位置直接从原文章截取，不使用 `tokenizer.decode()`。这样能
保留原来的中文格式，避免答案中出现多余空格。

## 六、验证与提交

```python
accuracy = sum(
    prediction == question["answer_text"]
    for question, prediction in zip(dev_questions, predictions)
) / len(dev_questions)
```

作业使用 Exact Match。只有预测答案和正确答案完全相同才算答对，多字或少字
都会被判错。

```python
if accuracy > best_accuracy:
    model.save_pretrained("outputs/best_model")
    tokenizer.save_pretrained("outputs/best_model")
```

每个 epoch 结束后，在验证集上计算准确率，并保存目前效果最好的模型。测试
时只需要加载 `best_model`。

```python
with open("result.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Answer"])

    for question, answer in zip(test_questions, predictions):
        writer.writerow([question["id"], answer.replace(",", "")])
```

最后，把 4,957 个测试问题的答案写入 `result.csv`。Kaggle 只需要上传这个
文件，不需要上传模型、代码或数据集。

## 七、总结

这次作业的重点不是从头实现 Transformer，而是学习如何微调预训练 BERT。
其中最重要的部分是长文章的滑动窗口、答案 token 位置转换，以及预测后的
答案筛选。模型本身很强，但如果窗口切分或后处理有问题，最终准确率仍然会很低。
