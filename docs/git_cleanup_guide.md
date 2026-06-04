# 🔧 Git 仓库清理 & .gitignore 修复流程

> 适用场景：项目已经 push 到 GitHub 后，发现有不该上传的文件（IDE 配置、数据集、模型权重、缓存等）需要从远程仓库中清除。

---

## 🎯 本文档解决什么问题

| 问题 | 后果 |
|:---|:---|
| `.idea/` 被上传 | 泄露个人 IDE 配置，污染 PR diff |
| `__pycache__/` 被上传 | 二进制垃圾文件，每次运行都变 |
| 76MB `train.csv` 在仓库里 | clone 极慢，GitHub 限制 100MB |
| `*.pt` / `*.model` 在仓库里 | 大文件膨胀，且可重新训练生成 |

---

## 📋 完整修复流程（4 步）

### Step 1：创建/完善 .gitignore

在项目根目录创建 `.gitignore`，分类写入忽略规则：

```gitignore
# ====================== IDE 配置文件 ======================
.idea/
.vscode/
*.iml
.DS_Store
Thumbs.db

# ====================== Python 运行时文件 ======================
__pycache__/
*.py[cod]
venv/
.venv/
*.egg-info/

# ====================== 训练产出 ======================
*.pt
*.pth
*.model
*.h5
logs/
runs/

# ====================== 数据集 ======================
data/*
!data/.gitkeep        # 保留目录结构，忽略数据文件

# ====================== 环境变量（含密钥） ======================
.env
.env.local
```

> **关键原则**：`.gitignore` 只对**未追踪（untracked）** 的文件生效，已经 `git add` 过的文件需要 Step 2 手动移除。

---

### Step 2：从 Git 追踪中移除已有文件

```
git rm --cached 只删追踪，不删本地文件
     ↑
  关键参数！不加会连本地文件一起删掉
```

```bash
# 逐个移除（推荐，精确控制）
git rm --cached -r .idea/
git rm --cached -r __pycache__/
git rm --cached data/train.csv
git rm --cached digital.model

# 验证：本地文件还在，但 git 不再追踪
ls data/train.csv        # ✅ 文件还在
git ls-files             # ✅ 列表中已没有
```

**`git rm --cached` vs `git rm`：**

| 命令 | 本地文件 | Git 追踪 |
|:---|:---|:---|
| `git rm --cached` | ✅ 保留 | ❌ 停止追踪 |
| `git rm`（无 --cached） | ❌ 删除 | ❌ 停止追踪 |

---

### Step 3：验证清理结果

```bash
# ① 查看暂存区（确认哪些被标记删除）
git status

# ② 查看仍有追踪的文件（只应该看到源码）
git ls-files
# ✅ 期望结果：只有 .py / .md / .gitignore / data/.gitkeep

# ③ 确认 gitignore 已生效（!! 前缀 = 被忽略）
git status --ignored --short
```

---

### Step 4：提交并推送

```bash
git add .gitignore data/.gitkeep
git commit -m "chore: add .gitignore and remove tracked binaries/datasets"
git push origin main
```

---

## 🧪 自检清单

推送后，在 GitHub 仓库页面对比确认：

- [ ] `.idea/` 目录已消失
- [ ] `__pycache__/` 目录已消失
- [ ] `data/train.csv` 已消失，但 `data/.gitkeep` 存在
- [ ] `*.pt` / `*.model` 等大文件已消失
- [ ] `.gitignore` 文件存在
- [ ] 源码文件（`.py`）正常显示
- [ ] `git clone` 后能直接运行（只需手动放入数据集）

---

## 🚨 常见坑

### 坑1：加了 .gitignore 但 git status 还能看到

> **原因**：文件在加 .gitignore **之前**就已经 `git add` 了。
>
> **解决**：执行 Step 2 的 `git rm --cached`。

### 坑2：`data/` 目录在 status 中显示为 untracked

> **原因**：目录中有文件未被任何 gitignore 规则匹配（如无后缀名文件 `nn_sample`）。
>
> **解决**：在 .gitignore 中用通配规则（如 `data/*`）覆盖所有文件，`!data/.gitkeep` 排外保留目录。

### 坑3：commit 历史中仍有大文件，clone 还是很慢

> **原因**：`git rm` 只删当前版本，历史记录中文件依然存在。
>
> **解决**（谨慎操作）：
> ```bash
> # 使用 BFG Repo-Cleaner 或 git filter-branch 清理历史
> # 注意：这会改写历史，协作项目需全员配合
> java -jar bfg.jar --delete-files "*.csv" --delete-files "*.pt" .
> git reflog expire --expire=now --all
> git gc --prune=now --aggressive
> git push --force
> ```

### 坑4：忘记 `!data/.gitkeep`，data 目录丢失

> Git 不追踪空目录。如果 `data/*` 全被忽略且没有 `.gitkeep`，clone 后 `data/` 目录不会创建。
>
> **解决**：总是配合 `!data/.gitkeep` 保留目录骨架。

---

## 📐 本项目实际执行记录

```
digitRecognition 清理前后对比
════════════════════════════════════════════════

清理前（追踪中 20 个文件）:
├── .idea/           (7 文件) ← IDE 配置
├── common/
│   ├── __init__.py
│   ├── __pycache__/ (2 文件) ← Python 缓存
│   └── load_data.py
├── data/
│   ├── train.csv    (76 MB)  ← 数据集
│   ├── nn_example.pt         ← 模型权重
│   └── nn_sample             ← 模型样本
├── digital.model             ← 训练产物
├── main.py                   ← 空模板
├── 01~04 脚本 + README

清理后（追踪中 9 个文件）:
├── .gitignore                ← 新增
├── 01_sklearn_softmax.py
├── 02_pytorch_mlp_inference.py
├── 03_pytorch_mlp_training.py
├── 04_pytorch_resnet_training.py
├── README.md
├── common/
│   ├── __init__.py
│   └── load_data.py
└── data/
    └── .gitkeep              ← 新增，保留目录

移除 13 个文件，新增 2 个文件，净减 11 个追踪文件
```

---

## 🔗 相关资源

- [GitHub 官方 .gitignore 模板](https://github.com/github/gitignore) — Python 项目推荐模板
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) — 清理 Git 历史中的大文件
- [git-scm 文档：gitignore](https://git-scm.com/docs/gitignore)
