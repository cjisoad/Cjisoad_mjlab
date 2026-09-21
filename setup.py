"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Keep the simulation stack together: mjlab 1.2 uses Warp's legacy context API.
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco==3.5.0",
    "mujoco-warp==3.5.0",
    "warp-lang==1.12.0",
    # RSL-RL 5.0.1 passes the deprecated wandb.Settings(start_method=...).
    "wandb==0.22.3",
    "scipy",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
