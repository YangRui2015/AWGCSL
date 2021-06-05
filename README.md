# Weighted Goal-conditioned Supervised Learning (WGCSL)
WGCSL is a simple but effective algorithm for both online and offline multi-goal Reinforcement Learning via weighted supervised learning.

<div style="text-align: center;">
<img src="pics/offline_random.png" height=250 >
</div>


## Requirements
python3.6+, tensorflow, gym, mujoco, mpi4py

## Installation
- Clone the repo and cd into it:

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Environments: Point2DLargeEnv-v1, Point2D-FourRoom-v1, FetchReach-v1, SawyerReachXYZEnv-v1, Reacher-v2, SawyerDoor-v0.

WGCSL: 
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode supervised --log_path ~/${path_name} --su_method exp_adv_clip10 
```

GCSL:
```bash
python -m  wgcsl.run  --env=Point2DLargeEnv-v1  --num_env 1 --mode supervised
```

GCSL + Discount Relabeling Weight:
```bash
python -m  wgcsl.run  --env=Point2DLargeEnv-v1  --num_env 1 --mode supervised --su_method gamma
```

GCSL + Goal-conditioned Exponential Advantage Weight:
```bash
python -m  wgcsl.run  --env=Point2DLargeEnv-v1  --num_env 1 --mode supervised --su_method exp_adv
```

