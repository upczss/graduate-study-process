# Git 的常用指令

这篇笔记记录现阶段最常用的 Git 指令。重点不是背命令，而是知道每条命令在什么场景下使用。

---

## 1. 查看 Git 是否安装成功

### `git --version`

```bash
git --version
```

作用：查看当前电脑是否安装 Git，以及 Git 的版本号。

使用场景：刚安装完 Git 后，用它检查是否安装成功。

---

## 2. 配置用户名和邮箱

Git 第一次使用前，需要配置用户名和邮箱。它们会出现在你的提交记录里。

### `git config --global user.name`

```bash
git config --global user.name "你的名字"
```

示例：

```bash
git config --global user.name "zss"
```

作用：设置 Git 提交时显示的用户名。

使用场景：第一次安装 Git 后配置。

### `git config --global user.email`

```bash
git config --global user.email "你的邮箱"
```

示例：

```bash
git config --global user.email "your_email@example.com"
```

作用：设置 Git 提交时绑定的邮箱。

使用场景：第一次安装 Git 后配置，建议使用 GitHub 绑定的邮箱。

### 查看当前配置

```bash
git config --global --list
```

作用：查看已经配置好的用户名、邮箱等信息。

使用场景：不确定自己有没有配置成功时使用。

---

## 3. 文件夹相关基础命令

这些不是 Git 专属命令，但在 Git Bash 里经常用。

### `pwd`

```bash
pwd
```

作用：查看当前所在目录。

使用场景：不知道自己现在在哪个文件夹时使用。

### `ls`

```bash
ls
```

作用：查看当前目录下有哪些文件和文件夹。

使用场景：进入一个文件夹后，查看里面的内容。

### `cd`

```bash
cd 文件夹名
```

示例：

```bash
cd learn-git
```

作用：进入某个文件夹。

使用场景：想进入项目目录时使用。

### 返回上一级目录

```bash
cd ..
```

作用：返回当前目录的上一级。

使用场景：进错文件夹，或者想回到上一层目录时使用。

### 创建文件夹

```bash
mkdir 文件夹名
```

示例：

```bash
mkdir learn-git
```

作用：创建一个新文件夹。

使用场景：想新建一个练习项目时使用。

---

## 4. 初始化 Git 仓库

### `git init`

```bash
git init
```

作用：把当前普通文件夹变成 Git 仓库。

使用场景：你自己新建了一个项目文件夹，想让 Git 开始管理它。

示例流程：

```bash
mkdir learn-git
cd learn-git
git init
```

注意：

执行 `git init` 后，这个文件夹里会出现一个隐藏的 `.git` 文件夹，它用来保存 Git 的版本信息。

---

## 5. 查看当前仓库状态

### `git status`

```bash
git status
```

作用：查看当前文件状态，比如哪些文件被修改了、哪些文件还没有提交。

使用场景：任何时候不知道下一步该干什么，都可以先输入它。

常见结果：

```text
Untracked files
```

表示有新文件，Git 还没有跟踪。

```text
Changes not staged for commit
```

表示文件修改了，但还没有加入暂存区。

```text
Changes to be committed
```

表示文件已经加入暂存区，可以提交。

```text
nothing to commit, working tree clean
```

表示当前没有需要提交的内容，工作区是干净的。

---

## 6. 添加到暂存区

### 添加单个文件

```bash
git add 文件名
```

示例：

```bash
git add note.md
```

作用：把指定文件加入暂存区，准备提交。

使用场景：只想提交某一个文件时使用。

### 添加所有变化

```bash
git add .
```

作用：把当前目录下所有新增、修改的文件加入暂存区。

使用场景：学习阶段最常用，适合一次性提交当前所有修改。

注意：

使用 `git add .` 前最好先执行：

```bash
git status
```

确认没有把不想提交的文件加进去。

---

## 7. 提交到本地仓库

### `git commit -m`

```bash
git commit -m "提交说明"
```

示例：

```bash
git commit -m "添加 Git 学习笔记"
```

作用：把暂存区里的内容正式保存成一次版本记录。

使用场景：完成一个小阶段的修改后，比如写完一篇笔记、完成一个功能、修复一个错误。

推荐提交说明写法：

```bash
git commit -m "添加 Markdown 语法笔记"
git commit -m "更新 Python 学习计划"
git commit -m "修复训练代码路径错误"
```

不推荐：

```bash
git commit -m "111"
git commit -m "随便改改"
git commit -m "更新"
```

---

## 8. 查看提交历史

### `git log`

```bash
git log
```

作用：查看详细提交历史。

使用场景：想知道之前提交过哪些版本时使用。

### 简洁查看提交历史

```bash
git log --oneline
```

作用：用一行显示一个提交记录，更简洁。

使用场景：学习阶段更推荐这个，信息清楚不复杂。

示例输出：

```text
a1b2c3d 添加 Git 学习笔记
b4c5d6e 初始化项目
```

---

## 9. 查看文件修改内容

### `git diff`

```bash
git diff
```

作用：查看工作区里具体改了什么。

使用场景：改完文件后，提交前想检查自己到底改了哪些内容。

### 查看暂存区里的修改

```bash
git diff --staged
```

作用：查看已经 `git add` 进暂存区、准备提交的内容。

使用场景：提交前最后检查一次。

---

## 10. 分支相关命令

分支可以理解为项目的不同路线。你现阶段先掌握基础即可。

### 查看当前分支

```bash
git branch
```

作用：查看本地有哪些分支，以及当前在哪个分支。

使用场景：不确定自己是否在 `main` 分支时使用。

### 创建新分支

```bash
git branch 分支名
```

示例：

```bash
git branch test
```

作用：创建一个新分支。

使用场景：想尝试一个新功能，但不想影响主分支。

### 切换分支

```bash
git switch 分支名
```

示例：

```bash
git switch test
```

作用：切换到指定分支。

使用场景：想去另一个分支继续修改时使用。

### 创建并切换到新分支

```bash
git switch -c 分支名
```

示例：

```bash
git switch -c notes-update
```

作用：创建一个新分支，并立刻切换过去。

使用场景：之后做项目时，开发新功能常用。

### 切回 main 分支

```bash
git switch main
```

作用：回到主分支。

使用场景：完成分支实验后，回到主线。

---

## 11. 连接 GitHub 远程仓库

远程仓库就是 GitHub 上的仓库。

### 查看远程仓库

```bash
git remote -v
```

作用：查看当前本地仓库连接了哪个 GitHub 仓库。

使用场景：不确定有没有连接远程仓库时使用。

### 添加远程仓库

```bash
git remote add origin 仓库地址
```

示例：

```bash
git remote add origin https://github.com/用户名/仓库名.git
```

作用：把本地仓库和 GitHub 仓库连接起来。

使用场景：本地已经有仓库，想上传到 GitHub 时使用。

注意：

`origin` 是远程仓库的常用默认名字。

---

## 12. 上传到 GitHub

### 第一次上传

```bash
git push -u origin main
```

作用：把本地 `main` 分支上传到 GitHub，并建立默认关联。

使用场景：第一次把本地仓库上传到 GitHub 时使用。

### 后续上传

```bash
git push
```

作用：把本地新提交的内容上传到 GitHub。

使用场景：以后每次 `commit` 后，想同步到 GitHub 时使用。

常用流程：

```bash
git status
git add .
git commit -m "更新学习笔记"
git push
```

---

## 13. 从 GitHub 下载或同步

### 克隆仓库

```bash
git clone 仓库地址
```

示例：

```bash
git clone https://github.com/用户名/仓库名.git
```

作用：把 GitHub 上的仓库完整下载到本地。

使用场景：第一次下载别人的项目，或者在新电脑上下载自己的项目。

### 拉取远程更新

```bash
git pull
```

作用：从 GitHub 拉取最新内容并合并到本地。

使用场景：远程仓库有更新，本地想同步时使用。

建议：

在开始写代码或笔记前，可以先执行：

```bash
git pull
```

---

## 14. 撤销相关命令

撤销命令要谨慎使用。现阶段先掌握安全的几条。

### 取消暂存

```bash
git restore --staged 文件名
```

示例：

```bash
git restore --staged note.md
```

作用：把文件从暂存区拿出来，但不会删除你的修改。

使用场景：不小心 `git add` 了某个文件，但还不想提交它。

### 撤销工作区修改

```bash
git restore 文件名
```

示例：

```bash
git restore note.md
```

作用：撤销某个文件还没有提交的修改，让它回到上一次提交后的状态。

使用场景：文件被自己改乱了，想恢复到最近一次提交的版本。

注意：

这个命令会丢掉当前未提交的修改，使用前要确认。

---

## 15. 删除文件并让 Git 记录删除

### `git rm`

```bash
git rm 文件名
```

示例：

```bash
git rm old_note.md
```

作用：删除文件，并让 Git 记录这次删除。

使用场景：确定某个文件不需要了，想从项目里删除。

后续还需要提交：

```bash
git commit -m "删除旧笔记"
```

---

## 16. 常用完整流程

### 场景一：新建本地仓库

```bash
mkdir learn-git
cd learn-git
git init
git status
```

使用场景：自己从零开始新建一个 Git 项目。

### 场景二：修改文件后提交

```bash
git status
git add .
git commit -m "更新学习笔记"
git status
```

使用场景：最常用，本地保存一次版本。

### 场景三：提交并上传 GitHub

```bash
git status
git add .
git commit -m "更新学习笔记"
git push
```

使用场景：本地修改完成后，上传到 GitHub 备份。

### 场景四：第一次连接 GitHub 并上传

```bash
git remote add origin https://github.com/用户名/仓库名.git
git push -u origin main
```

使用场景：本地仓库第一次上传到 GitHub。

### 场景五：从 GitHub 下载项目

```bash
git clone https://github.com/用户名/仓库名.git
```

使用场景：下载自己的云端仓库，或者下载别人的开源项目。

---

## 17. 现阶段最应该熟练的命令

如果你现在只背一组，就背下面这些：

```bash
git status
git add .
git commit -m "提交说明"
git log --oneline
git push
git pull
git clone 仓库地址
```

对应含义：

| 命令 | 含义 |
|---|---|
| `git status` | 查看当前状态 |
| `git add .` | 添加所有修改到暂存区 |
| `git commit -m "说明"` | 保存一次本地版本 |
| `git log --oneline` | 查看简洁提交历史 |
| `git push` | 上传到 GitHub |
| `git pull` | 从 GitHub 同步更新 |
| `git clone 仓库地址` | 下载远程仓库 |

---

## 18. Git 学习口诀

```text
先 status 看状态
再 add 加暂存
再 commit 存本地
最后 push 传 GitHub
```

日常最常用流程：

```bash
git status
git add .
git commit -m "更新内容"
git push
```

#Git #GitHub #学习笔记
