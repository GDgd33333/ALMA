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

# 导入各种注册表，用于模块化管理
from learners import REGISTRY as le_REGISTRY  # 学习器注册表
from runners import REGISTRY as r_REGISTRY  # 运行器注册表
from controllers import REGISTRY as mac_REGISTRY  # 多智能体控制器注册表
from envs import s_REGISTRY  # 环境注册表
from components.episode_buffer import ReplayBuffer  # 经验回放缓冲区
from components.transforms import OneHot  # OneHot编码转换


def run(config, console_logger, wandb_run):
    """主运行函数，负责整个训练/评估流程的协调"""
    # 检查参数合理性
    config = args_sanity_check(config, console_logger)

    # 将配置字典转换为命名空间对象，便于访问
    # config = {"use_cuda": True}变成 args.use_cuda # True
    args = SN(**config)
    # 根据CUDA可用性设置设备
    args.device = "cuda" if args.use_cuda else "cpu"

    # 设置日志记录器
    logger = Logger(console_logger)

    # 如果不是评估模式且不使用保存回放，则设置wandb日志
    if not (args.evaluate or args.save_replay) and args.use_wandb:
        logger.setup_wandb()

    # 打印实验参数信息
    console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config,
                                       indent=4,
                                       width=1)
    console_logger.info("\n\n" + experiment_params + "\n")

    # 配置tensorboard日志记录器
    # 生成唯一的时间戳token
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    args.unique_token = unique_token
    # 如果使用tensorboard，设置相关目录
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.tb_dirname)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # 运行和训练
    run_sequential(args=args, logger=logger)

    # 如果wandb运行存在，则结束它
    if wandb_run is not None:
        wandb_run.finish()

    # 清理工作
    print("Exiting Main")

    # 停止所有线程
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)  # 等待线程结束，最多等待1秒
            print("Thread joined")

    print("Exiting script")

    # 确保框架真正退出
    os._exit(os.EX_OK)


def load_run(args, wb_run, learner, runner, logger, pi_only=False):
    """从wandb运行中加载模型"""
    timesteps = []  # 存储时间步列表
    timestep_to_load = 0  # 要加载的时间步

    # 获取wandb运行中的所有文件
    files = wb_run.files()
    # 从文件名中提取时间步信息
    timesteps = set(int(f.name.split('_')[0]) for f in files if f.name.endswith('.th'))

    # 根据load_step参数选择要加载的时间步
    if args.load_step == 0:
        # 选择最大时间步
        timestep_to_load = max(timesteps)
    else:
        # 选择最接近load_step的时间步
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    # 筛选出需要加载的文件
    files_to_load = [f for f in files if f.name.startswith(str(timestep_to_load))]
    # 如果只需要策略网络，只加载agent.th文件
    if pi_only:
        files_to_load = [f for f in files_to_load if 'agent.th' in f.name]
    # 生成唯一ID用于创建临时目录
    unique_id = uuid.uuid4().hex[:6].lower()
    model_dir = os.path.join(f'./eval_models', unique_id)
    os.makedirs(model_dir)  # 创建模型目录
    # 从wandb恢复文件到本地
    for file in files_to_load:
        wandb.restore(file.name, run_path='/'.join(wb_run.path),
                      replace=True, root=model_dir)
    # 构建模型路径
    model_path = os.path.join(model_dir, f"{timestep_to_load}_")
    logger.console_logger.info(f"Loading model from run: {wb_run.name}")
    # 加载模型到学习器和运行器
    learner.load_models(model_path, evaluate=args.evaluate, pi_only=pi_only)
    runner.t_env = timestep_to_load

def get_wandb_runs(checkpoint_name, wandb_api, args):
    """根据检查点名称获取wandb运行"""
    # 根据环境类型选择评估指标
    if 'sc2' in args.env:
        metric = 'eval_env_metrics/battle_won_mean'  # StarCraft II环境使用战斗胜利率
    elif args.env == 'ff':
        metric = 'eval_env_metrics/solved_mean'  # 其他环境使用解决率
    else:
        raise Exception('Need to define best model metric for this environment')
    # 查询匹配的实验运行
    exp_runs = wandb_api.runs(
        path=f'{args.wb_entity}/task-allocation',
        filters={'config.name': checkpoint_name},
        order=f'-summary_metrics.{metric}')  # 按指标降序排列
    assert len(exp_runs) > 0, "No matching runs found"
    return exp_runs

def evaluate_sequential(args, runner, logger):
    """顺序执行评估"""
    vw = None  # 视频写入器
    # 如果指定了视频路径，设置视频录制
    if args.video_path is not None:
        os.makedirs(dirname(args.video_path), exist_ok=True)
        vid_basename_split = splitext(basename(args.video_path))
        if vid_basename_split[1] == '.mp4':
            vid_basename = ''.join(vid_basename_split)
        else:
            vid_basename = ''.join(vid_basename_split) + '.mp4'
        vid_filename = join(dirname(args.video_path), vid_basename)
        # 创建视频写入器
        vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                fps=args.fps, codec='h264', quality=10)

    res_dict = {}  # 结果字典

    # 根据评估模式确定运行次数
    if args.eval_all_scen:
        # 评估所有场景
        if 'sc2' in args.env:
            dict_key = 'scenarios'
        else:
            raise Exception("Environment (%s) does not incorporate multiple scenarios")
        n_runs = len(args.env_args['scenario_dict'][dict_key])
    elif args.eval_n_task_range != "":
        # 评估指定任务数量范围
        min_n_tasks, max_n_tasks = args.eval_n_task_range.split("-")
        n_task_range = list(range(int(min_n_tasks), int(max_n_tasks) + 1))
        n_runs = len(n_task_range)
    else:
        # 默认单次运行
        n_runs = 1
    # 计算测试批次数量
    n_test_batches = max(1, args.test_nepisode // runner.batch_size)

    all_subtask_infos = []  # 存储所有子任务信息
    # 执行多次评估运行
    for i in range(n_runs):
        logger.console_logger.info(f"Running evaluation on setting {i + 1}/{n_runs}")
        # 设置运行参数
        run_args = {'test_mode': True, 'vid_writer': vw,
                    'test_scen': True}
        if args.eval_all_scen:
            run_args['index'] = i  # 设置场景索引
        elif args.eval_n_task_range != "":
            run_args['n_tasks'] = n_task_range[i]  # 设置任务数量
        # 执行测试批次
        for _ in range(n_test_batches):
            _, new_subtask_infos = runner.run(**run_args)
            all_subtask_infos.extend(new_subtask_infos)
        # 获取当前统计信息
        curr_stats = dict((k, v[-1][1]) for k, v in logger.stats.items())
        # 根据评估模式存储结果
        if args.eval_all_scen:
            curr_scen = args.env_args['scenario_dict'][dict_key][i]
            # 假设唯一智能体集合代表唯一场景
            if 'sc2' in args.env:
                scen_str = "-".join("%i%s" % (count, name[:3]) for count, name in sorted(curr_scen[0], key=lambda x: x[1]))
            else:
                scen_str = "".join(curr_scen[0])
            res_dict[scen_str] = curr_stats
        elif args.eval_n_task_range != "":
            res_dict[n_task_range[i]] = curr_stats
        else:
            res_dict.update(curr_stats)

    # 关闭视频写入器
    if vw is not None:
        vw.close()

    # 如果设置了保存回放，则保存
    if args.save_replay:
        runner.save_replay()
    return res_dict, all_subtask_infos


def run_sequential(args, logger):
    """顺序运行训练/评估流程"""
    # 初始化运行器以获取环境信息
    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    # 为特定环境设置场景字典
    if args.env in ['sc2custom', 'sc2multiarmy', 'ff']:
        rs = RandomState(0)  # 固定随机种子，排除场景随机性对结果的影响
        args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)

    # 如果使用启发式任务分配，设置相应标志
    if args.hier_agent["task_allocation"] == 'heuristic':
        args.env_args['heuristic_alloc'] = True

    # 为StarCraft II环境设置额外单位标签数量
    if ('sc2custom' == args.env or 'sc2multiarmy' == args.env):
        args.env_args['n_extra_tags'] = args.n_extra_units

    # 设置混合器子任务条件 subtask_cond是mask，mixer_subtask_cond也是mask
    if args.mixer_subtask_cond is None:
        args.mixer_subtask_cond = args.agent["subtask_cond"]
    
    # 创建运行器
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 设置方案和组
    env_info = runner.get_env_info()  # 获取环境信息
    args.n_agents = env_info["n_agents"]  # 智能体数量
    args.n_actions = env_info["n_actions"]  # 动作数量
    if not args.entity_scheme:
        # 默认/基础方案
        args.state_shape = env_info["state_shape"]

        scheme = {
            "state": {"vshape": env_info["state_shape"]},  # 状态形状
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # 观察形状
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # 动作
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},  # 可用动作
            "reward": {"vshape": (1,)},  # 奖励
            "terminated": {"vshape": (1,), "dtype": th.uint8},  # 终止标志
            "reset": {"vshape": (1,), "dtype": th.uint8},  # 重置标志
            "ep_num": {"vshape": (1,), "dtype": th.long, "episode_const": True}  # 回合编号
        }
        groups = {
            "agents": args.n_agents  # 智能体组
        }
        args.pre_transition_items = ['state', 'obs', 'avail_actions']  # 转换前项目
        args.post_transition_items = ['reward', 'terminated', 'reset']  # 转换后项目
        if 'masks' in env_info:
            # 标识观察/状态空间中每个实体对应部分的掩码
            args.obs_masks, args.state_masks = env_info['masks']
        if 'unit_dim' in env_info:
            args.unit_dim = env_info['unit_dim']
    else:
        # 实体方案
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]
        scheme = {
            "entities": {"vshape": env_info["entity_shape"], "group": "entities"},  # 实体
            "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8},  # 观察掩码
            "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},  # 实体掩码
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # 动作
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},  # 可用动作
            "reward": {"vshape": (1,)},  # 奖励
            "terminated": {"vshape": (1,), "dtype": th.uint8},  # 终止标志
            "reset": {"vshape": (1,), "dtype": th.uint8},  # 重置标志
            "t_added": {"vshape": (1,), "dtype": th.long, "episode_const": True}  # 添加时间
        }
        args.pre_transition_items = ['entities', 'obs_mask', 'entity_mask', 'avail_actions']  # 转换前项目
        args.post_transition_items = ['reward', 'terminated', 'reset']  # 转换后项目
        groups = {
            "agents": args.n_agents,  # 智能体组
            "entities": args.n_entities  # 实体组
        }
        # 如果使用多任务，添加任务相关字段
        if args.multi_task:
            args.n_tasks = env_info["n_tasks"]
            scheme["entity2task_mask"] = {"vshape": args.n_tasks, "group": "entities", "dtype": th.uint8}  # 实体到任务掩码
            scheme["task_mask"] = {"vshape": args.n_tasks, "dtype": th.uint8}  # 任务掩码
            args.pre_transition_items += ["entity2task_mask", "task_mask"]
            args.post_transition_items += ["task_rewards", "tasks_terminated"]  # 任务奖励和终止标志
            scheme["task_rewards"] = {"vshape": args.n_tasks}  # 任务奖励
            scheme["tasks_terminated"] = {"vshape": args.n_tasks, "dtype": th.uint8}  # 任务终止标志
            # 如果使用分层智能体的任务分配或COPA
            if args.hier_agent["task_allocation"] is not None or args.hier_agent["copa"]:
                scheme["hier_decision"] = {"vshape": (1,), "dtype": th.uint8}  # 分层决策
                args.pre_transition_items += ['hier_decision']

    # 设置预处理
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])  # 动作的OneHot编码
    }

    # 创建经验回放缓冲区
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device,
                          efficient_store=args.buffer_opt_mem,
                          max_traj_len=args.max_traj_len)

    # 设置多智能体控制器
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 为运行器设置方案
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 创建学习器
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # 如果使用CUDA，将学习器移到GPU
    if args.use_cuda:
        learner.cuda()

    # 处理检查点加载
    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
        assert not (args.checkpoint_run_name != ""
                    and args.checkpoint_unique_id != ""), (
                        "Can only specify one of checkpoint_run_name or checkpoint_unique_id"
                    )
        wandb_api = wandb.Api()
        pi_runs = None
        if args.checkpoint_run_name != "":
            # 根据运行名称获取实验运行
            exp_runs = get_wandb_runs(args.checkpoint_run_name, wandb_api, args)
            if args.pi_checkpoint_run_name != "":
                if args.pi_checkpoint_run_name == "same":
                    assert len(exp_runs) > 1, "Need multiple seeds to swap policies"
                    pi_runs = list(exp_runs)[1:] + list(exp_runs)[:1]
                else:
                    pi_runs = get_wandb_runs(args.pi_checkpoint_run_name, wandb_api, args)
            if args.eval_all_models:
                runs = exp_runs  # 评估所有模型
            else:
                # 选择最终性能最好的模型
                runs = [exp_runs[0]]
        elif args.checkpoint_unique_id != "":
            # 根据唯一ID获取运行
            runs = [wandb_api.run(f'{args.wb_entity}/task-allocation/{args.checkpoint_unique_id}')]

    # 如果指定了检查点或使用启发式AI，执行评估
    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "" or args.env_args.get('heuristic_ai', False):
        if args.evaluate or args.save_replay:
            # 设置评估结果保存路径
            if args.eval_path is not None:
                os.makedirs(dirname(args.eval_path), exist_ok=True)
                eval_basename_split = splitext(basename(args.eval_path))
                if eval_basename_split[1] == '.json':
                    eval_basename = ''.join(eval_basename_split)
                else:
                    eval_basename = ''.join(eval_basename_split) + '.json'
                eval_filename = join(dirname(args.eval_path), eval_basename)

            if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
                results = []
                # 对每个运行进行评估
                for i, wb_run in enumerate(runs):
                    logger.console_logger.info(f"Evaluating model {i + 1}/{len(runs)}")
                    load_run(args, wb_run, learner, runner, logger)
                    # 如果指定了策略检查点，加载策略
                    if pi_runs is not None:
                        pi_run = pi_runs[i % len(pi_runs)]
                        load_run(args, pi_run, learner, runner, logger, pi_only=True)
                    res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)
                    results.append(res_dict)

                # 保存评估结果
                if args.eval_path is not None:
                    if args.eval_sep:
                        write_struct = all_subtask_infos
                    else:
                        write_struct = results
                    with open(eval_filename, 'w') as f:
                        json.dump(write_struct, f)
            else:
                # 使用启发式方法
                res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)
            runner.close_env()  # 关闭环境
            logger.print_stats_summary()  # 打印统计摘要
            return
    #---------------------------------------------------------------------- 
    #-------------新增加，打印args对象的所有参数（程序实际使用的参数），mixer_subtask_cond会变化成和subtask_cond一样的‘mask’----------------------------------
    # 打印args对象的所有参数（程序实际使用的参数）
    logger.console_logger.info("=" * 80)
    logger.console_logger.info("ACTUAL RUNTIME PARAMETERS (args object):")
    logger.console_logger.info("=" * 80)
    actual_params = pprint.pformat(vars(args), indent=4, width=1)
    logger.console_logger.info(actual_params)
    logger.console_logger.info("=" * 80)
    #---------------------------------------------------------------------- 
    #---------------------------------------------------------------------- 
    # 开始训练
    episode = 0  # 回合计数器
    last_test_T = -args.test_interval - 1  # 上次测试时间步
    last_log_T = 0  # 上次日志时间步
    model_save_time = 0  # 模型保存时间

    start_time = time.time()  # 开始时间
    last_time = start_time  # 上次时间

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # 主训练循环
    while runner.t_env <= args.t_max:

        # 运行一个完整回合
        episode_batch, _ = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)  # 将回合数据插入缓冲区

        # 如果缓冲区可以采样，进行训练
        if buffer.can_sample(args.batch_size):
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)  # 从缓冲区采样

                # 截断批次到只包含填充的时间步
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                # 如果样本不在指定设备上，移动到该设备
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # 训练学习器
                learner.train(episode_sample, runner.t_env, episode)

                # 如果使用AQL任务分配，进行额外的分配训练
                if args.hier_agent["task_allocation"] in ["aql"]:
                    filters = {}
                    if args.hier_agent['decay_old'] > 0:
                        cutoff = args.hier_agent['decay_old']
                        filters['t_added'] = lambda t_added: (runner.t_env - t_added) <= cutoff
                    if not buffer.can_sample(args.batch_size, filters=filters):
                        continue

                    alloc_episode_sample = buffer.sample(args.batch_size, filters=filters)

                    # 截断批次到只包含填充的时间步
                    max_ep_t = alloc_episode_sample.max_t_filled()
                    alloc_episode_sample = alloc_episode_sample[:, :max_ep_t]

                    if alloc_episode_sample.device != args.device:
                        alloc_episode_sample.to(args.device)

                    elif args.hier_agent["task_allocation"] == "aql":
                        learner.alloc_train_aql(alloc_episode_sample, runner.t_env, episode)

        # 定期执行测试运行
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            # 执行测试运行
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 定期保存模型
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
            # 学习器应该处理保存/加载 -- 将actor保存/加载委托给mac，
            # 使用适当的文件名处理critics、优化器状态
            learner.save_models(save_path_base)

        episode += args.batch_size_run  # 更新回合计数

        # 定期记录日志
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()  # 关闭环境
    logger.console_logger.info("Finished Training")


# TODO: 清理这个函数
def args_sanity_check(config, console_logger):
    """检查参数合理性"""
    # 设置CUDA标志
    # config["use_cuda"] = True # 尽可能使用cuda！
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        console_logger.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    # 确保测试回合数不小于批次运行大小
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        # 将测试回合数调整为批次运行大小的整数倍，确保测试时能完整运行多个批次
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # 检查分层智能体配置的一致性
    # 如果配置了分层智能体的任务分配或智能体的子任务条件，则两者必须同时配置
    # hier_agent["task_allocation"]：None，'heuristic'，'aql'，'random'，'random_fixed'
    # agent["subtask_cond"]：None，'mask'，'full_obs'

    if (config["hier_agent"]["task_allocation"] is not None
            or config["agent"]["subtask_cond"] is not None):
        # 断言：子任务条件类型和任务分配必须同时指定，确保配置的完整性和一致性
        assert (config["agent"]["subtask_cond"] is not None
                and config["hier_agent"]["task_allocation"] is not None), (
            "Subtask-conditioning type and task allocation must be specified together")

    # 注释掉的断言，用于检查并行模式和重放缓冲区的兼容性
    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # 注释掉的COMA特定检查
    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
