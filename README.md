# Weighted Goal-conditioned Supervised Learning (WGCSL)
WGCSL is a simple but effective algorithm for both online and offline multi-goal Reinforcement Learning via weighted supervised learning.

<!-- <div style="text-align: center;">
<img src="pics/model-based-relabeling.png" height=250 >
</div> -->


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

WGCSL: (valid methods-- 'gamma_01_adv', 'gamma_tanh_adv', 'gamma_exp_adv', 'exp_adv',  '01_adv', 'tanh_adv', 'gamma')
```bash
python3 -m  awgcsl.run  --env=${envname}  --num_env 1 --mode supervised --log_path ~/${path_name} --su_method ${method} 
```

DDPG:
```bash
python -m  awgcsl.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 --noher True --log_path=~/logs/point/ --save_path=~/logs/ddpg/point/model/
```
HER:
```bash
python -m  awgcsl.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 
```
GCSL:
```bash
python -m  awgcsl.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 --mode supervised
```
