# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 22.04 
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 550 or later  

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 Activate the Virtual Environment

```bash
conda activate unitree_rl_mjlab
```

---

## 2. Installing

### 2.1 Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 Install Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

All other dependencies are specified in the setup.py file.
Navigate to the project root directory and install them with:

```bash
cd unitree_rl_mjlab
python -m pip install -e .
```

The installer pins the compatible simulation stack to `mjlab==1.2.0`,
`mujoco==3.5.0`, `mujoco-warp==3.5.0`, and `warp-lang==1.12.0`, and installs
`scipy` for terrain imports. Upgrading MuJoCo or Warp independently can break
APIs used by this version of mjlab. Re-run the command above to update an
existing environment to these constraints.

W&B is pinned to `wandb==0.22.3` to support the deprecated
`wandb.Settings(start_method="thread")` call in the current RSL-RL logger.
Versions that remove this option fail with
`ValidationError: start_method / Extra inputs are not permitted`.
To repair an existing environment, activate the Conda environment used for
training and run:

```bash
python -m pip install "wandb==0.22.3"
```

Then repeat the training command. This failure happens during W&B startup,
before training, and is unrelated to the robot task or velocity rewards.
No edits to `site-packages` or Pydantic downgrade are needed.

Check the Go2 rear-leg standing task and run a short GPU training check:

```bash
python scripts/train.py Unitree-Go2-RearStand --help
python scripts/train.py Unitree-Go2-RearStand --env.scene.num-envs 16 --agent.max-iterations 1
```

Start full training with:

```bash
python scripts/train.py Unitree-Go2-RearStand --env.scene.num-envs 4096
```

GPU 0 is selected by default. Explicit GPU lists use Python list syntax:
`--gpu-ids '[0]'` or `--gpu-ids '[0, 1]'`. Reduce the environment count if
GPU memory is insufficient. Checkpoints are saved under `logs/rsl_rl/go2_stand/`.

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.
