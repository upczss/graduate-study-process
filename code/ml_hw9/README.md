# HW9 Explainable AI

这份代码复用当前工作区中已经训练好的两个模型：

- HW3 Food-11 CNN：`../ml_hw3/best_model.pt`
- HW7 BERT QA：`../ml_hw7/outputs/best_model/`

不需要重新训练。所有可视化默认保存在 `outputs/`。

## 环境

```bash
conda activate mllearning
cd C:\Users\zss\OneDrive\Desktop\obsidian\code\ml_hw9
python -m pip install -r requirements.txt
```

如果 HW7 已经可以运行，通常只需要补装：

```bash
python -m pip install lime matplotlib
```

## CNN 可解释性

一次运行五种方法：

```bash
python run.py cnn --method all
```

为了先快速检查，可只用两张图片和较少计算量：

```bash
python run.py cnn --method saliency --image-indices 0,1
python run.py cnn --method smoothgrad --image-indices 0,1 --smooth-samples 10
python run.py cnn --method integrated-gradients --image-indices 0,1 --ig-steps 20
python run.py cnn --method filter --image-indices 0,1 --filter-index 0
python run.py cnn --method lime --image-indices 0,1 --lime-samples 200
```

默认图片来自验证集，索引是 `0,1,...,9`。由于原始 Colab 已经失效，官方指定的十张图片索引无法确认；若之后找到索引，可这样替换：

```bash
python run.py cnn --method all --image-indices 12,35,48,71,96,120,155,201,260,318
```

卷积层名称和 filter 编号也可以修改：

```bash
python run.py cnn --method filter --layer cnn_layer5.block.0 --filter-index 10
```

## BERT 可解释性

一次运行注意力、Embedding PCA 和相似度分析：

```bash
python run.py bert --method all --example-index 0
```

分别运行：

```bash
python run.py bert --method attention --attention-layer -1 --attention-head 0
python run.py bert --method embedding --pca-layers 0,6,12
python run.py bert --method similarity --token-a 福 --token-b 州
```

如果不指定 `token-a` 和 `token-b`，程序会自动选择前两个普通 token。可先查看 `outputs/bert/example.json` 中的 token 列表，再选择想比较的词。

## 输出文件

CNN：

- `lime.png`
- `saliency.png`
- `smoothgrad.png`
- `filter_*.png`
- `filter_maximization_*.png`
- `integrated_gradients.png`
- `predictions.csv`

BERT：

- `attention.png`
- `embedding_pca.png`
- `similarity.json`
- `example.json`

这些文件只用于观察模型，不是 Kaggle 提交文件。
