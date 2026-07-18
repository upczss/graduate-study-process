# git
## 简介

Git 是一个**版本管理工具**。是分布式管理系统

方便管理文件并且可以记录修改过程

- 写完一部分代码，存一个档
- 笔记改坏了，可以回到以前版本
- 每次修改了什么，都能查到
- 多台电脑之间可以同步
- 多个人可以一起协作一个项目

常用概念：

|概念|意思|
|---|---|
|仓库 repository|被 Git 管理的项目文件夹|
|commit|一次存档|
|add|把文件加入本次存档|
|push|上传到 GitHub|
|pull|从 GitHub 拉取更新|
|clone|从 GitHub 下载项目|

最常用流程：

```
git status
git add .
git commit -m "更新学习笔记"
git push
```

使用方式：GUI 命令行和编辑器插件，主要是使用命令行

## 创建仓库
1. 从本地创建自己的仓库

```
mkdir learn-git
cd learn-git
git init
ls -a(查看是否有.git文件)
```
不要随意修改.git文件里面的内容这也是为什么要把git隐藏起来的原因

2. 从github上克隆仓库（远程仓库）后面补充

## 工作区域与文件状态

### 工作区域
Git 主要有 **三个工作区域**，你可以把它理解成“从修改文件到正式存档”的三个步骤。

```
工作区 → 暂存区 → 本地仓库
```

---

**1. 工作区 Working Directory**

就是你电脑里正在编辑的项目文件夹。

比如你现在的：

```
learn-git
```

你在里面新建、修改、删除文件，这些变化一开始都在**工作区**。

例子：

```
touch note.md
```

这时 `note.md` 只是出现在工作区里，Git 还没有正式准备保存它。

---

**2. 暂存区 Staging Area**

暂存区可以理解成：**准备提交的清单**。

你用：

```
git add note.md
```

或者：

```
git add .
```

就是把文件变化放进暂存区。

意思是告诉 Git：

```
这次提交，我准备保存这些文件变化。
```

---

**3. 本地仓库 Local Repository**

本地仓库就是 Git 真正保存历史版本的地方。

你用：

```
git commit -m "添加学习笔记"
```

就是把暂存区里的内容正式保存成一个版本。

这一步才算真正“存档”。

---

**完整流程**

```
touch note.md
git status
git add note.md
git status
git commit -m "添加 note 笔记"
git status
```

对应关系是：

```
touch note.md
      ↓
文件出现在工作区

git add note.md
      ↓
文件进入暂存区

git commit -m "添加 note 笔记"
      ↓
文件进入本地仓库，形成一次版本记录
```

---

**再加上 GitHub 的话**

如果你还要上传到 GitHub，就多一个远程仓库：

```
工作区 → 暂存区 → 本地仓库 → 远程仓库 GitHub
```

对应命令：

```
git add .
git commit -m "更新笔记"
git push
```

---

最简单记法：

|区域|作用|命令|
|---|---|---|
|工作区|你正在改文件的地方|直接编辑文件|
|暂存区|准备提交的文件清单|`git add`|
|本地仓库|正式保存版本|`git commit`|
|远程仓库|GitHub 上的云端仓库|`git push`|

### 文件状态
![[Pasted image 20260708211922.png]]

| 状态         | 含义          | 下一步          |
| ---------- | ----------- | ------------ |
| Untracked  | 新文件，Git 还没管 | `git add`    |
| Modified   | 已跟踪文件被修改了   | `git add`    |
| Staged     | 已放入暂存区      | `git commit` |
| Unmodified | 没有变化，很干净    | 继续写代码/笔记     |
## 添加和提交文件
1. 首先是创建一个文件
2. 然后用git status来检查状态
3. 用git add （文件名）来上传到暂存区域
4. 用git commit 来上传暂存区的文件到仓库里面，已上传的文件在使用git status的时候就不会在显示出来了因为已经保管到仓库里面了。
5. git add .表示的是添加全部的文件到暂存区，git add \*.txt 

## git回退功能

git reset的三种用法
git reset --soft   保留暂存区和工作区
git reset --hard  既不保留暂存区也不保存工作区
git reset --mixed 保留工作区但不保留暂存区

`git reset` 的核心作用是：**把 Git 的某个区域退回到指定状态**。

它主要影响三个地方：

```
工作区：你正在编辑的文件
暂存区：git add 后准备提交的内容
本地仓库：git commit 后形成的版本记录
```

`git reset` 最常见有三种用法：

```
git reset --soft
git reset --mixed
git reset --hard
```

区别在于：**回退 commit 时，保不保留你的代码修改。**

---

**先准备一个例子**

假设你有 3 次提交：

```
C 第三次提交：写了 PyTorch 笔记
B 第二次提交：写了 Git 笔记
A 第一次提交：创建项目
```

查看提交历史：

```
git log --oneline
```

可能看到：

```
c333333 写了 PyTorch 笔记
b222222 写了 Git 笔记
a111111 创建项目
```

现在你想从 `C` 回到 `B`。

---

**1. `git reset --soft`**

```
git reset --soft b222222
```

效果：

```
本地仓库回到 B
C 这次提交被取消
C 里面的修改还在
而且这些修改仍然在暂存区
```

适合场景：

你刚刚 commit 了，但发现提交说明写错了，或者想重新组织提交。

例子：

```
git commit -m "随便写的提交信息"
git reset --soft HEAD~1
git commit -m "添加 PyTorch 学习笔记"
```

解释：

```
HEAD~1
```

表示回退到上一个提交。

`--soft` 很安全，因为你的代码不会丢。

---

**2. `git reset --mixed`**

```
git reset --mixed b222222
```

或者直接写：

```
git reset b222222
```

因为 `--mixed` 是默认模式。

效果：

```
本地仓库回到 B
C 这次提交被取消
C 里面的修改还在工作区
但不在暂存区
```

适合场景：

你 commit 了，但想取消这次提交，并重新选择哪些文件要提交。

例子：

```
git reset HEAD~1
```

效果：

```
最近一次 commit 被撤销
文件修改还保留
但是需要重新 git add
```

然后你可以：

```
git status
git add note.md
git commit -m "只提交 note.md"
```

这个也比较安全，因为修改还在。

---

**3. `git reset --hard`**

```
git reset --hard b222222
```

效果：

```
本地仓库回到 B
C 这次提交被取消
C 里面的修改也被删除
工作区也回到 B 的样子
```

这个要小心。

适合场景：

你确定最近的修改完全不要了，想让项目彻底回到某个版本。

例子：

```
git reset --hard HEAD~1
```

意思是：

```
删除最近一次提交
同时删除这次提交里的所有文件修改
```

如果这些修改没有推送、没有备份，可能就找不回来了。

新手阶段：**少用 `--hard`。**

---

**4. `git reset HEAD 文件名`**

这个非常常用，也比较安全。

```
git reset HEAD note.md
```

作用：

```
把 note.md 从暂存区拿出来
但是不会删除文件内容
```

比如你执行了：

```
git add .
```

但突然发现 `note.md` 不想提交，可以：

```
git reset HEAD note.md
```

现在更推荐的新命令是：

```
git restore --staged note.md
```

效果类似。

---

**5. `git reset HEAD~1`**

这个也很常见。

```
git reset HEAD~1
```

作用：

```
撤销最近一次 commit
保留文件修改
把修改放回工作区
```

使用场景：

你刚刚提交了：

```
git commit -m "更新"
```

但发现提交太乱，想重新提交：

```
git reset HEAD~1
git status
git add 想提交的文件
git commit -m "更清楚的提交说明"
```

---

**三种模式对比**

|命令|commit 回退|暂存区|工作区代码|
|---|---|---|---|
|`git reset --soft HEAD~1`|回退|保留在暂存区|保留|
|`git reset --mixed HEAD~1`|回退|取消暂存|保留|
|`git reset --hard HEAD~1`|回退|清空|删除修改|

简单理解：

```
--soft：撤销 commit，但保留 add
--mixed：撤销 commit，也撤销 add，但保留代码
--hard：撤销 commit、撤销 add、删除代码修改
```

---

**具体学习场景示例**

假设你在学习 Git，做了这些操作：

```
echo "Git reset 学习" > reset.md
git add reset.md
git commit -m "添加 reset 笔记"
```

后来你发现提交信息太随便，想改提交信息。

用：

```
git reset --soft HEAD~1
git commit -m "添加 Git reset 用法笔记"
```

这样文件不会丢，只是重新提交。

如果你发现这次提交里混进了两个文件：

```
git reset HEAD~1
git add reset.md
git commit -m "添加 reset 笔记"
```

这样可以重新选择要提交的文件。

如果你发现这个文件完全不想要：

```
git reset --hard HEAD~1
```

这会把最近一次提交和文件修改都删掉。这个慎用。

---

**一句话总结**

`git reset` 是用来“回退版本或取消暂存”的命令。

新手最常用、最安全的是：

```
git reset HEAD~1
```

撤销最近一次提交，但保留修改。

以及：

```
git reset HEAD 文件名
```

取消暂存某个文件。

最危险的是：

```
git reset --hard
```

因为它可能直接丢掉你的修改。

## gitdiff查找差异

git diff 代表的是查找工作区与暂存区的差异，如果相同则不会显示内容

`git diff` 的作用是：**查看文件具体改了哪里。**

它一般在 `git add` 或 `git commit` 之前使用，用来检查自己这次到底改了什么。

---

**1. 查看还没 add 的修改**

```
git diff
```

使用场景：

你修改了文件，但还没有执行 `git add`，想看看自己改了哪些内容。即只会比较工作区与暂存区的差别

例子：

你修改了 `note.md`，然后输入：

```
git diff
```

Git 会显示：

```
- 原来的内容
+ 新增或修改后的内容
```

其中：

```
- 表示删除或修改前的内容
+ 表示新增或修改后的内容
```

---

**2. 查看已经 add 的修改**

如果你已经执行了：

```
git add .
```

这时候再执行：

```
git diff
```

可能什么都不显示。

因为普通的 `git diff` 只看“还没 add 的修改”。

这时要用：

```
git diff --staged
```

或者：

```
git diff --cached
```

作用：查看已经放进暂存区、准备提交的内容。

---

**3. 查看某个文件的修改**

```
git diff 文件名
```

例子：

```
git diff note.md
```

作用：只查看 `note.md` 这个文件改了什么。

---

**4. 常用流程**

提交前建议这样做：

```
git status
git diff
git add .
git diff --staged
git commit -m "更新笔记"
```

意思是：

```
git status       看哪些文件变了
git diff         看还没 add 的修改
git add .        放进暂存区
git diff --staged 看准备提交的内容
git commit       正式提交
```

---

**你现阶段记这三个就够**

```
git diff
```

查看还没 `add` 的修改。

```
git diff --staged
```

查看已经 `add`、准备 `commit` 的修改。

```
git diff 文件名
```

只查看某个文件的修改。

`git diff HEAD~ HEAD`
表示比较的是最新版本与上一个版本的差别，HEAD是指向最新版本的指针HEAD~代表的上面几个版本，波浪号后面什么也不加代表的是上一个版本，加一个数字2代表的是上面两个版本

## 删除文件
git rm 文件名，就可以把文件从暂存区和工作区中一起删除，然后再提交一下即可
![[Pasted image 20260708235805.png]]

## github管理代码
### 如何将本地仓库上传到远程仓库

可以。你现在要做的是：

```
暑假学习文件夹 → 变成本地 Git 仓库 → 连接 GitHub → push 上传
```

注意：你前面文件夹好像叫 **暑假学习**，不是 **暑期学习**。你先以实际文件夹名为准。

**第一步：进入你的学习文件夹**

在 Git Bash 输入：

```
cd ~/OneDrive/Desktop/obsidian/暑假学习
```

如果你的文件夹真的叫 `暑期学习`，就用：

```
cd ~/OneDrive/Desktop/obsidian/暑期学习
```

然后确认当前位置：

```
pwd
ls
```

---

**第二步：初始化本地仓库**

```
git init
git branch -M main
```

这一步的意思是：把当前 `暑假学习` 文件夹变成 Git 本地仓库，并把主分支命名为 `main`。

---

**第三步：把文件加入本地仓库**

先查看状态：

```
git status
```

然后添加所有文件：

```
git add .
```

提交：

```
git commit -m "第一次提交：添加暑假学习笔记"
```

到这里，你的文件已经进入**本地仓库**了。

---

**第四步：连接 GitHub 仓库**

你需要先在 GitHub 网站上创建一个空仓库，比如：

```
graduate-study-process
```

然后复制 SSH 地址，应该类似：

```
git@github.com:upczss/graduate-study-process.git
```

回到 Git Bash，执行：

```
git remote add origin git@github.com:upczss/graduate-study-process.git
```

如果提示：

```
remote origin already exists
```

说明已经连接过了，那就改成：

```
git remote set-url origin git@github.com:upczss/graduate-study-process.git
```

查看是否连接成功：

```
git remote -v
```

---

**第五步：上传到 GitHub**

```
git push -u origin main
```

以后再更新笔记，只需要：

```
git status
git add .
git commit -m "更新学习笔记"
git push
```

你现阶段完整流程就是：

```
cd ~/OneDrive/Desktop/obsidian/暑假学习
git init
git branch -M main
git add .
git commit -m "第一次提交：添加暑假学习笔记"
git remote add origin git@github.com:upczss/graduate-study-process.git
git push -u origin main
```

如果你的 GitHub 仓库已经不是空的，比如里面有 README，`push` 可能会报冲突。到时候把报错贴给我，我带你一步步处理。


### 将远程仓库中的文件拉取到本地仓库之中

你要把 GitHub 上别人新提交的文件同步到本地，用：

```
git pull
```

一般流程是：

```
cd ~/OneDrive/Desktop/obsidian/暑假学习
git status
git pull
```

**建议先 `git status`**，看你本地有没有还没提交的修改。

---

如果你本地很干净，看到类似：

```
nothing to commit, working tree clean
```

那就直接：

```
git pull
```

Git 会把 GitHub 上的新文件拉到你的本地。

---

如果你本地也改了文件，还没提交，建议先提交自己的修改：

```
git status
git add .
git commit -m "保存本地学习笔记修改"
git pull
```

然后如果没冲突，再：

```
git push
```

---

你日常可以记这个顺序：

**开始学习前：**

```
git pull
```

先同步 GitHub 上的最新内容。

**学习结束后：**

```
git status
git add .
git commit -m "更新学习笔记"
git push
```

把自己的新内容传上去。

---
如果说远程仓库上已经被修改过了内容的话，此时应该先git pull然后再git push

## 分支 
多个开发员可以在自己的分支上面开发程序，独立开发而不影响主线工程
git branch (name of branch)创建分支
git switch (name of branch)切换分支
git merge dev(这个是在main分支下执行的语句，代表的是把dev分支合并到main分支上面)分支合并之后dev分支还存在，需要手动的删除一下dev分支，代码是：
git branch -d dev（删除已经合并的分支）
git branch -D dev（删除未合并的分支）
git log --graph --oneline --decorate --all(用于显示分支图)

### 如何解决合并冲突
当两个branch同时改了同一个文件的同一行代码的时候此时git会发生合并冲突，此时需要我们进行手动合并

# 上传
以后在 Git Bash 上传整个 Obsidian 文件夹，按下面操作即可。

第一次先进入仓库目录：

```
cd "/c/Users/zss/OneDrive/Desktop/obsidian"
```

查看有哪些修改：

```
git status
```

把所有新增、修改和删除暂存起来：

```
git add .
```

创建一次提交，双引号里写本次修改内容：

```
git commit -m "更新学习笔记"
```

上传到 GitHub：

```
git push
```

完整流程就是：

```
cd "/c/Users/zss/OneDrive/Desktop/obsidian"
git status
git add .
git commit -m "更新学习笔记"
git push
```

几个常见情况：

- 如果 `git commit` 显示 `nothing to commit`，说明没有新修改，不需要上传。
- 如果 `git push` 显示 `Everything up-to-date`，说明 GitHub 已是最新状态。
- 如果换电脑或 GitHub 上先有了新修改，上传前先运行：

```
git pull --rebase
```

- 不要再在内部文件夹运行 `git init`，例如不要在 `暑假学习`、`learn-git` 或 `code` 中初始化仓库，否则又会产生嵌套仓库。
- 始终在最外层的 `obsidian` 文件夹里执行 Git 命令。

你也可以通过下面这条命令确认当前位置是否正确：

```
git rev-parse --show-toplevel
```

正确结果应该是：

```
C:/Users/zss/OneDrive/Desktop/obsidian
```