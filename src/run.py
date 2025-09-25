import datetime
from functools import partial
from math import ceil
import imageio
import wandb
import os
import pprint
import time
import json
import threading
import uuid
import torch as th
from numpy.random import RandomState
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath, basename, join, splitext

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from envs import s_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(config, console_logger, wandb_run):
    # check args sanity
    config = args_sanity_check(config, console_logger)

    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(console_logger)

    if not (args.evaluate or args.save_replay) and args.use_wandb:
        logger.setup_wandb()

    console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config,
                                       indent=4,
                                       width=1)
    console_logger.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.tb_dirname)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # Run and train
    run_sequential(args=args, logger=logger)

    if wandb_run is not None:
        wandb_run.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def load_run(args, wb_run, learner, runner, logger, pi_only=False):
    timesteps = []
    timestep_to_load = 0

    files = wb_run.files()
    timesteps = set(int(f.name.split('_')[0]) for f in files if f.name.endswith('.th'))

    if args.load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    files_to_load = [f for f in files if f.name.startswith(str(timestep_to_load))]
    if pi_only:
        files_to_load = [f for f in files_to_load if 'agent.th' in f.name]
    unique_id = uuid.uuid4().hex[:6].lower()
    model_dir = os.path.join(f'./eval_models', unique_id)
    os.makedirs(model_dir)
    for file in files_to_load:
        wandb.restore(file.name, run_path='/'.join(wb_run.path),
                      replace=True, root=model_dir)
    model_path = os.path.join(model_dir, f"{timestep_to_load}_")
    logger.console_logger.info(f"Loading model from run: {wb_run.name}")
    learner.load_models(model_path, evaluate=args.evaluate, pi_only=pi_only)
    runner.t_env = timestep_to_load

def get_wandb_runs(checkpoint_name, wandb_api, args):
    if 'sc2' in args.env:
        metric = 'eval_env_metrics/battle_won_mean'
    elif args.env == 'ff':
        metric = 'eval_env_metrics/solved_mean'
    else:
        raise Exception('Need to define best model metric for this environment')
    exp_runs = wandb_api.runs(
        path=f'{args.wb_entity}/task-allocation',
        filters={'config.name': checkpoint_name},
        order=f'-summary_metrics.{metric}')
    assert len(exp_runs) > 0, "No matching runs found"
    return exp_runs

def evaluate_sequential(args, runner, logger):
    vw = None
    if args.video_path is not None:
        os.makedirs(dirname(args.video_path), exist_ok=True)
        vid_basename_split = splitext(basename(args.video_path))
        if vid_basename_split[1] == '.mp4':
            vid_basename = ''.join(vid_basename_split)
        else:
            vid_basename = ''.join(vid_basename_split) + '.mp4'
        vid_filename = join(dirname(args.video_path), vid_basename)
        vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                fps=args.fps, codec='h264', quality=10)

    res_dict = {}

    if args.eval_all_scen:
        if 'sc2' in args.env:
            dict_key = 'scenarios'
        else:
            raise Exception("Environment (%s) does not incorporate multiple scenarios")
        n_runs = len(args.env_args['scenario_dict'][dict_key])
    elif args.eval_n_task_range != "":
        min_n_tasks, max_n_tasks = args.eval_n_task_range.split("-")
        n_task_range = list(range(int(min_n_tasks), int(max_n_tasks) + 1))
        n_runs = len(n_task_range)
    else:
        n_runs = 1
    n_test_batches = max(1, args.test_nepisode // runner.batch_size)

    all_subtask_infos = []
    for i in range(n_runs):
        logger.console_logger.info(f"Running evaluation on setting {i + 1}/{n_runs}")
        run_args = {'test_mode': True, 'vid_writer': vw,
                    'test_scen': True}
        if args.eval_all_scen:
            run_args['index'] = i
        elif args.eval_n_task_range != "":
            run_args['n_tasks'] = n_task_range[i]
        for _ in range(n_test_batches):
            _, new_subtask_infos = runner.run(**run_args)
            all_subtask_infos.extend(new_subtask_infos)
        curr_stats = dict((k, v[-1][1]) for k, v in logger.stats.items())
        if args.eval_all_scen:
            curr_scen = args.env_args['scenario_dict'][dict_key][i]
            # assumes that unique set of agents is a unique scenario
            if 'sc2' in args.env:
                scen_str = "-".join("%i%s" % (count, name[:3]) for count, name in sorted(curr_scen[0], key=lambda x: x[1]))
            else:
                scen_str = "".join(curr_scen[0])
            res_dict[scen_str] = curr_stats
        elif args.eval_n_task_range != "":
            res_dict[n_task_range[i]] = curr_stats
        else:
            res_dict.update(curr_stats)

    if vw is not None:
        vw.close()

    if args.save_replay:
        runner.save_replay()
    return res_dict, all_subtask_infos

'''
Runner 负责和环境交互，按 pre/post_transition_items 把每步数据打包成 EpisodeBatch。
ReplayBuffer 用 scheme + groups + preprocess 定义“怎么存、怎么取、如何 one-hot 化”等。
MAC 是策略前向（多智能体参数共享或实体输入等），Learner 负责优化（值网络/混合器/策略头）。
训练主循环是：rollout → buffer.insert → sample → learner.train → 定期eval → 定期save。
若用 AQL 分配器，会额外按“近期数据”再做一次专门的训练（时间过滤 t_added）
'''
def run_sequential(args, logger):
    # Init runner so we can get env info
    # 初始化 runner 前，对环境参数做前置整理 
    # 若环境字典里显式给了 entity 模式开关，则用之；否则默认关闭
    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    # 某些环境族（自定义 SC2、多军团、消防员 ff）需要根据场景名生成 scenario_dict
    if args.env in ['sc2custom', 'sc2multiarmy', 'ff']:
        rs = RandomState(0)  # 固定随机种子，生成可复现实验的场景
        args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)

    # 若任务分配策略选择启发式（heuristic），通知环境走启发式分配路径
    if args.hier_agent["task_allocation"] == 'heuristic':
        args.env_args['heuristic_alloc'] = True

    # 对自定义的 sc2 环境，可能需要给到“额外标签”的数量
    if ('sc2custom' == args.env or 'sc2multiarmy' == args.env):
        args.env_args['n_extra_tags'] = args.n_extra_units

    # 如果没有显式给 mixer 的子任务条件（subtask_cond），则与 agent 的一致
    if args.mixer_subtask_cond is None:
        args.mixer_subtask_cond = args.agent["subtask_cond"]
    
    # 实例化 Runner（负责与环境交互、收集轨迹）
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # —— 根据环境信息配置数据 schema、group、preprocess —— 
    env_info = runner.get_env_info()       # 取出环境报告的 shape、上限等
    args.n_agents = env_info["n_agents"]   # 智能体数
    args.n_actions = env_info["n_actions"] # 动作数
    
    if not args.entity_scheme:
        # 非 entity 模式：使用“state/obs”扁平方案
        args.state_shape = env_info["state_shape"]

        # ReplayBuffer 存储的字段定义（每个时间步要写什么）
        scheme = {
            "state": {"vshape": env_info["state_shape"]},         # 全局状态
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},   # 每个 agent 的局部观测
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # 离散动作索引
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},  # 可行动作掩码
            "reward": {"vshape": (1,)},                           # 标量奖励
            "terminated": {"vshape": (1,), "dtype": th.uint8},    # 回合终止标记
            "reset": {"vshape": (1,), "dtype": th.uint8},         # 环境是否 reset（有些环境需要）
            "ep_num": {"vshape": (1,), "dtype": th.long, "episode_const": True}    # 回合编号（对整条轨迹常量）
        }
        groups = {
            "agents": args.n_agents   # 指定“agents”这条轴的长度（多智能体维度）
        }
        # 定义“转移前/后”需要从环境采集的键（用于 runner 收集 & buffer 写入）
        args.pre_transition_items = ['state', 'obs', 'avail_actions']
        args.post_transition_items = ['reward', 'terminated', 'reset']

        # 若环境提供了掩码（将大向量中特定切片映射到不同实体），存起来供模型使用
        if 'masks' in env_info:
            args.obs_masks, args.state_masks = env_info['masks']
        # 单位（entity/unit）维度可选信息
        if 'unit_dim' in env_info:
            args.unit_dim = env_info['unit_dim']
    else:
        # entity 模式：输入为“实体列表”，每步数量固定为 n_entities，带掩码表明哪些有效
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]

        scheme = {
            "entities": {"vshape": env_info["entity_shape"], "group": "entities"},    # (n_entities, feat_dim)
            "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8}, # 每个 agent 的可见实体掩码
            "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},     # 哪些实体在当前步有效
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},         # 离散动作
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "reset": {"vshape": (1,), "dtype": th.uint8},
            "t_added": {"vshape": (1,), "dtype": th.long, "episode_const": True}      # 样本加入缓冲的时间戳（用于时间衰减过滤）
        }
        # entity 模式下，前/后转移需要收集的字段
        args.pre_transition_items = ['entities', 'obs_mask', 'entity_mask', 'avail_actions']
        args.post_transition_items = ['reward', 'terminated', 'reset']
        groups = {
            "agents": args.n_agents,
            "entities": args.n_entities
        }

         # 多任务（multi_task）扩展：额外的任务掩码与任务奖励
        if args.multi_task:
            args.n_tasks = env_info["n_tasks"]
            scheme["entity2task_mask"] = {"vshape": args.n_tasks, "group": "entities", "dtype": th.uint8}  # 实体到任务的映射掩码
            scheme["task_mask"] = {"vshape": args.n_tasks, "dtype": th.uint8}                              # 当前步哪些任务有效
            args.pre_transition_items += ["entity2task_mask", "task_mask"]
            args.post_transition_items += ["task_rewards", "tasks_terminated"]  # 每个任务的奖励与终止
            scheme["task_rewards"] = {"vshape": args.n_tasks}
            scheme["tasks_terminated"] = {"vshape": args.n_tasks, "dtype": th.uint8}
            # 若使用分层/分配（层次智能体）则加一个决策标志位
            if args.hier_agent["task_allocation"] is not None or args.hier_agent["copa"]:
                scheme["hier_decision"] = {"vshape": (1,), "dtype": th.uint8}
                args.pre_transition_items += ['hier_decision']

    # 预处理：将离散动作索引转 one-hot（很多算法/混合器需要）
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 创建经验回放缓冲区
    buffer = ReplayBuffer(
        scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,  # 轨迹长度 = episode_limit+1（含终止步）
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,            # 可把存储放 CPU 节省显存
        efficient_store=args.buffer_opt_mem,                              # 内存优化开关
        max_traj_len=args.max_traj_len                                    # 可选：用于截断长轨迹
    )

    # —— 创建多智能体控制器（策略）并绑定给 runner ——
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)  # e.g., BasicMAC / entity 版本 MAC
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # —— Learner 负责参数更新（值函数/混合器/优化器等） ——
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()  # 把网络移动到 GPU


    # —— 处理从 wandb checkpoint 加载（可选） ——
    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
        assert not (args.checkpoint_run_name != "" and args.checkpoint_unique_id != ""), \
            "两种 checkpoint 指定方式只能二选一"
        wandb_api = wandb.Api()
        pi_runs = None
        if args.checkpoint_run_name != "":
            exp_runs = get_wandb_runs(args.checkpoint_run_name, wandb_api, args)  # 根据 run_name 拉取候选
            if args.pi_checkpoint_run_name != "":
                # 只加载策略部分的 run（支持“same”轮换不同种子）
                if args.pi_checkpoint_run_name == "same":
                    assert len(exp_runs) > 1, "需要多个 seed 才能做轮换加载"
                    pi_runs = list(exp_runs)[1:] + list(exp_runs)[:1]
                else:
                    pi_runs = get_wandb_runs(args.pi_checkpoint_run_name, wandb_api, args)
            # 若 eval_all_models=True，则对所有 run 评估；否则只取最优的一个
            if args.eval_all_models:
                runs = exp_runs
            else:
                runs = [exp_runs[0]]
        elif args.checkpoint_unique_id != "":
            # 通过唯一 id 精确指定一个 wandb run
            runs = [wandb_api.run(f'{args.wb_entity}/task-allocation/{args.checkpoint_unique_id}')]

    # —— 只评估/保存回放（不训练）的分支 —— 
    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "" or args.env_args.get('heuristic_ai', False):
        if args.evaluate or args.save_replay:
            # 若要求把评估结果写到文件，先准备输出路径/文件名
            if args.eval_path is not None:
                os.makedirs(dirname(args.eval_path), exist_ok=True)
                eval_basename_split = splitext(basename(args.eval_path))
                if eval_basename_split[1] == '.json':
                    eval_basename = ''.join(eval_basename_split)
                else:
                    eval_basename = ''.join(eval_basename_split) + '.json'
                eval_filename = join(dirname(args.eval_path), eval_basename)

            if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
                # 逐个加载 wandb run 的模型并评估
                results = []
                for i, wb_run in enumerate(runs):
                    logger.console_logger.info(f"Evaluating model {i + 1}/{len(runs)}")
                    load_run(args, wb_run, learner, runner, logger)  # 载入 value/mixer/actor 等
                    if pi_runs is not None:
                        # 只替换策略（π）权重做组合评估
                        pi_run = pi_runs[i % len(pi_runs)]
                        load_run(args, pi_run, learner, runner, logger, pi_only=True)
                    res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)
                    results.append(res_dict)

                # 写评估结果（可按任务拆分/或整体）
                if args.eval_path is not None:
                    if args.eval_sep:
                        write_struct = all_subtask_infos
                    else:
                        write_struct = results
                    with open(eval_filename, 'w') as f:
                        json.dump(write_struct, f)
            else:
                # 纯启发式 agent 的评估
                res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)

            runner.close_env()
            logger.print_stats_summary()
            return  # 评估完成直接退出

    # —— 正式进入训练主循环 ——
    episode = 0
    last_test_T = -args.test_interval - 1  # 上次测试时间步（初始化成足够小，保证第一次触发）
    last_log_T = 0                         # 上次打印日志的时间步
    model_save_time = 0                    # 上次保存模型的时间步

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    while runner.t_env <= args.t_max:

        # 每次收集“一个完整回合”的数据
        episode_batch, _ = runner.run(test_mode=False)  # 与环境交互，返回一个 EpisodeBatch
        buffer.insert_episode_batch(episode_batch)      # 写入回放缓冲

        # 当缓冲可采样时，执行若干次训练迭代
        if buffer.can_sample(args.batch_size):
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)   # 采样一个 batch 的若干回合片段

                # 只保留每条轨迹中已填充的时间步（去掉尾部空白）
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                # 把数据张量搬到目标设备（CPU/GPU）
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # 参数更新一步
                learner.train(episode_sample, runner.t_env, episode)

                # —— 若层次分配器是 AQL，还需要单独抽样“较新”的数据再训一次 —— 
                if args.hier_agent["task_allocation"] in ["aql"]:
                    filters = {}
                    if args.hier_agent['decay_old'] > 0:
                        cutoff = args.hier_agent['decay_old']
                        # 仅使用“近期加入”的数据（根据 t_added 做时间窗口过滤）
                        filters['t_added'] = lambda t_added: (runner.t_env - t_added) <= cutoff
                    if not buffer.can_sample(args.batch_size, filters=filters):
                        continue

                    alloc_episode_sample = buffer.sample(args.batch_size, filters=filters)

                    # 同样裁掉未填满的时间步
                    max_ep_t = alloc_episode_sample.max_t_filled()
                    alloc_episode_sample = alloc_episode_sample[:, :max_ep_t]

                    if alloc_episode_sample.device != args.device:
                        alloc_episode_sample.to(args.device)

                    elif args.hier_agent["task_allocation"] == "aql":
                        # 训练 AQL 分配器
                        learner.alloc_train_aql(alloc_episode_sample, runner.t_env, episode)

        # —— 按间隔做评估回合（不更新参数） ——
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)  # 仅前向评估性能

        # —— 按间隔或训练结束时保存模型 —— 
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or
                                model_save_time == 0 or
                                runner.t_env > args.t_max):
            model_save_time = runner.t_env
            if args.use_wandb:
                save_path_base = os.path.join(wandb.run.dir, "%i_" % (runner.t_env))
            else:
                save_path_base = os.path.join("results/models", args.unique_token, str(runner.t_env), "")
            os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(os.path.dirname(save_path_base)))
            # 交给 learner 自己序列化 actor/critic/优化器等
            learner.save_models(save_path_base)

        # 逻辑上“推进 episode 计数”（若一次 run 收集多回合，这里通常等于 batch_size_run）
        episode += args.batch_size_run

        # —— 打印训练统计 —— 
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # 训练结束，清理环境
    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, console_logger):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        console_logger.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    if (config["hier_agent"]["task_allocation"] is not None
            or config["agent"]["subtask_cond"] is not None):
        assert (config["agent"]["subtask_cond"] is not None
                and config["hier_agent"]["task_allocation"] is not None), (
            "Subtask-conditioning type and task allocation must be specified together")

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
