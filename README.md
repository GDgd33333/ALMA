# 飞书上：ALMA代码中有具体完整的运行命令说明


1）：运行save the city 环境
# 让当前 Shell 认识 conda（非登录 shell 时必需）
source /data/gu-di/miniconda3/etc/profile.d/conda.sh

# 激活pytorch 环境
conda activate pytorch

# 下载包：
pip install protobuf==3.20.3 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 一、开训练会话

# 1. 新建一个 tmux 会话 (叫 ALMAff)
tmux new -s ALMAff

# 2. 在会话里启动训练，并保存日志
tmux new -s ALMAsavethecity

# 进入会话后再手动：
source /data/gu-di/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
cd ~/ALMA-main/src


CUDA_VISIBLE_DEVICES=1 python3 main.py --env-config=ff --config=qmix_atten --agent.subtask_cond='mask' --hier_agent.task_allocation='aql' --epsilon_anneal_time=2000000 --use_tensorboard=True --save_model=True --save_model_interval=1000000 --hier_agent.action_length=5 2>&1 | tee ALMAsavethecity.log


tensorboard --logdir /data/gu-di/ALMA-main/results/tb_logs --port 6001 --host 127.0.0.1


# 如果不小心关掉了：
  tmux ls   #查看有哪些会话
  tmux attach -t ALMAff  # ALMAff 是会话名
# 删除会话
  tmux kill-session -t ALMAff

# 说明：
  # --env-config=ff:
    选择 SaveTheCity 环境（在本仓库里用缩写 ff 表示）。
  # --config=qmix_atten:
    选择低层控制算法的配置为 qmix_atten（带注意力的 QMIX 变体，便于多 agent 的信息聚合/混合）。
  # --agent.subtask_cond='mask'（ALMA 推荐项）:
    高层决定子任务后，低层 agent 接收被mask后的子任务条件。这能限制无关信息干扰、稳定训练。对照项：'full_obs' 表示低层看到完整观测（无遮罩）。
  # --hier_agent.task_allocation='aql'（ALMA 核心）:
    高层的“任务分配”模块采用 Allocation Q-Learning (AQL) 来把子任务分派给不同 agent 群组/个体，是 ALMA 的关键设计。对照项：'heuristic' 使用手工启发式分配（见下面的 --env_args.heuristic_style），或 --hier_agent.copa=True 切到 COPA 方法。
  # --epsilon_anneal_time=2000000（环境推荐）:
    探索率 epsilon 的线性退火步数；
  # --hier_agent.action_length=5（层级方法推荐）:
    高层动作的持续步长（也可理解为宏动作长度或分配策略的持有时间）。对分层方法（ALMA/COPA/启发式）仓库建议 5（StarCraft 则建议 3）。








# ALMA
Code for [*ALMA: Hierarchical Learning for Composite Multi-Agent Tasks*](https://openreview.net/forum?id=JUXn1vXcrLA) (Iqbal et al., NeurIPS 2022)

This code is built on the [public code release for REFIL](https://github.com/shariqiqbal2810/REFIL) which is built on the [PyMARL framework](https://github.com/oxwhirl/pymarl)

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@inproceedings{iqbal2022alma,
title={ALMA: Hierarchical Learning for Composite Multi-Agent Tasks},
author={Shariq Iqbal and Robby Costales and Fei Sha},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=JUXn1vXcrLA}
}
```

## Installation instructions

1. Install Docker
2. Install NVIDIA Docker if you want to use GPU (recommended)
3. Build the docker image using 
```bash
cd docker
./build.sh
```
4. Set up StarCraft II. If installed already on your machine just make sure `SC2PATH` is set correctly, otherwise run:
```bash
./install_sc2.sh
```
5. Make sure `SC2PATH` is set to the installation directory (`3rdparty/StarCraftII`)
6. Make sure `WANDB_API_KEY` is set if you want to use weights and biases

## Running experiments

Use the following command to run:
```bash
./run.sh <GPU> python3.7 src/main.py \
    --config=<alg> --env-config=<env> --scenario=<scen>
```
with the bracketed parameters replaced as follows:
* `<GPU>`: The index of the GPU you would like to run this experiment on
* `<alg>`: The low-level learning algorithm (choices are `qmix_atten` or `refil`)
* `<env>`: The environment
  * `ff`: SaveTheCity environment
  * `sc2multiarmy`: StarCraft environment
* `<scen>`: Specifies set of tasks in the environment (for StarCraft)
  * `6-8sz_maxsize4_maxarmies3_symmetric`: Stalkers and Zealots Symmetric
  * `6-8sz_maxsize4_maxarmies3_unitdisadvantage`: Stalkers and Zealots Disadvantage
  * `6-8MMM_maxsize4_maxarmies3_symmetric`: MMM Symmetric
  * `6-8MMM_maxsize4_maxarmies3_unitdisadvantage`: MMM Disadvantage

Method-Specific parameters:
* ALMA: Use `--agent.subtask_cond='mask'` and `--hier_agent.task_allocation='aql'`
* ALMA (No Mask):  `--agent.subtask_cond='full_obs'` and `--hier_agent.task_allocation='aql'`
* Heuristic Allocation: Use `--agent.subtask_cond='mask'` and `--hier_agent.task_allocation='heuristic'`
  * StarCraft (Dynamic):  `--env_args.heuristic_style='attacking-type-unassigned-diff'`
  * StarCraft (Matching): `--env_args.heuristic_style='type-unassigned-diff'`
* COPA: `--hier_agent.copa=True`

Environment-Specific hyperparameters:
* `SaveTheCity`
  * Use `--epsilon_anneal_time=2000000` for all methods
  * Use `--hier_agent.action_length=5` for hierarchical methods (allocation-based and COPA)
  * Use `--config=qmix_atten`
* `StarCraft`
  * Use `--hier_agent.action_length=3` for hierarchical methods (allocation-based and COPA)
  * Use `--config=refil`

Miscellaneous parameters:
* Weights and Biases: To use, make a project named "task-allocation" in weights and biases and include the following parameters in your runs. Make sure `WANDB_API_KEY` is set.
  * `--use-wandb=True`: Enables W&B logging,
  * `--wb-notes`: Notes associated with this experiment,
  * `--wb-tags` Specify list of tags separated by spaces
  * `--wb-entity` Specify W&B user or group name