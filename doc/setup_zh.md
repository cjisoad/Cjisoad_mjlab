# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 激活虚拟环境

```bash
conda activate unitree_rl_mjlab
```

---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 安装依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

我们将其余所需依赖放入 setup.py 文件中，
进入 unitree_rl_mjlab 项目根目录并安装：

```bash
cd unitree_rl_mjlab
python -m pip install -e .
```

安装配置固定了相互兼容的仿真依赖：`mjlab==1.2.0`、`mujoco==3.5.0`、
`mujoco-warp==3.5.0` 和 `warp-lang==1.12.0`，并补充地形模块导入所需的
`scipy`。单独升级 MuJoCo 或 Warp 可能破坏当前 mjlab 使用的接口。
已有环境也可以重新运行上面的安装命令，使版本符合这些约束。

W&B 固定为 `wandb==0.22.3`，兼容当前 RSL-RL 日志初始化使用的
`wandb.Settings(start_method="thread")`。部分新版 W&B 已移除此参数，
会报 `ValidationError: start_method / Extra inputs are not permitted`。
已有环境遇到该错误时，在训练使用的 Conda 环境中执行：

```bash
python -m pip install "wandb==0.22.3"
```

随后重新运行原训练命令。此错误发生在 W&B 初始化阶段，与机器人任务或
速度奖励无关；不需要修改 `site-packages` 或降级 Pydantic。

检查 Go2 后腿双足站立任务，并运行一次小规模 GPU 训练验证：

```bash
python scripts/train.py Unitree-Go2-RearStand --help
python scripts/train.py Unitree-Go2-RearStand --env.scene.num-envs 16 --agent.max-iterations 1
```

正式训练命令：

```bash
python scripts/train.py Unitree-Go2-RearStand --env.scene.num-envs 4096
```

默认使用 GPU 0。显式指定 GPU 时，列表必须使用 Python 列表语法：
`--gpu-ids '[0]'` 或 `--gpu-ids '[0, 1]'`。显存不足时可减少并行环境数量。
模型保存在 `logs/rsl_rl/go2_stand/` 下。

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。
