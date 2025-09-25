import copy
from components.episode_buffer import EpisodeBatch
from functools import partial
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from components.action_selectors import parse_avail_actions
import torch as th
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop, Adam

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args                   # 训练的所有超参数/配置（从命令行与 yaml 合并后的字典/对象）
        self.mac = mac                     # 多智能体控制器（Multi-Agent Controller），封装各 agent 的策略/价值网络
        self.logger = logger                # 日志记录器，用于定期 log scalar 等
        self.use_copa = self.args.hier_agent['copa']  # 是否启用 COPA 相关的层级/约束模块

        self.params = list(mac.parameters())     # 需要用“主优化器”训练的参数初始为 MAC 的参数
        if self.use_copa:
            self.params += list(self.mac.coach.parameters())  # 启用 COPA 时，把“教练/高层”网络的参数也并入主优化器
            if self.args.hier_agent['copa_vi_loss']:
                self.params += list(self.mac.copa_recog.parameters())  # 若开启 VI 正则（值迭代一致性），再并入识别/结构模块参数

        self.last_target_update_episode = 0      # 记录上一次“低层 target 网络”同步的 episode 计数（用于间隔更新）
        self.last_alloc_target_update_episode = 0 # 记录上一次“高层分配 target 网络”同步的 episode 计数

        self.mixer = None                        # QMIX/VDN 等“值函数混合器”，把各个体 Q 聚合为联合 Q_tot
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()          # VDN：各体 Q 简单相加
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)        # QMIX：受限单调性的可学习混合网络
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)    # Flex-QMIX：基于实体的可变体数混合（要求 entity_scheme=True）
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)  # 线性版本的 Flex-QMIX
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer)) 
            self.params += list(self.mixer.parameters())         # 将混合器参数加入主优化器训练
            self.target_mixer = copy.deepcopy(self.mixer)        # 创建混合器的 target 副本（延迟更新）

        # 主优化器：优化“self.params”里的所有模块（MAC、本体 mixer、以及可选 COPA 模块）
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # 如果高层任务分配使用 AQL（Allocation Q-Learning）
        if self.args.hier_agent["task_allocation"] == "aql":
            self.alloc_pi_params = list(mac.alloc_pi_params())   # 获取“分配策略 π”的参数列表（policy，用于输出任务分配）
            if self.args.hier_agent["alloc_opt"] == "rmsprop":
                OptClass = partial(RMSprop, alpha=args.optim_alpha)  # 高层可单独指定优化器（默认 RMSprop，复用 alpha）
            elif self.args.hier_agent["alloc_opt"] == "adam":
                OptClass = Adam                                   # 或者用 Adam
            else:
                raise Exception("Optimizer not recognized")       # 未知优化器选项

            # 分配策略 π 的优化器：学习把 critic 的偏好摊销/蒸馏到 policy（对应 alloc_amort_loss、entropy 正则等）
            self.alloc_pi_optimiser = OptClass(
                params=self.alloc_pi_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.weight_decay)

            self.alloc_q_params = list(mac.alloc_q_params())     # 获取“分配 Q-critic”的参数列表（评估任务分配的值）
            # 分配 Q 的优化器：用 TD 目标训练高层 critic（对应 alloc_q_loss）
            self.alloc_q_optimiser = OptClass(
                params=self.alloc_q_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.alloc_q_weight_decay)          # 注意：Q-critic 可以使用单独的 weight_decay

        # 复制一份 target MAC（包含 agent 网络与 action selector 等）
        # 虽然会重复一些非必要部分（比如动作选择器），但 deep copy 能保证结构一致，更新安全。
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # 日志打印的时间计数器（以 env step/t_env 为单位）：初始化为“上次打印时刻 = -间隔-1”，确保一开始就会打印一次
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_alloc_stats_t = -self.args.learner_log_interval - 1  # 高层分配相关日志的独立计时器

    def _get_mixer_ins(self, batch):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1],
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            mix_ins = {"entities": entities[:, :-1],
                       "entity_mask": batch["entity_mask"][:, :-1]}
            targ_mix_ins = {"entities": entities[:, 1:],
                            "entity_mask": batch["entity_mask"][:, 1:]}
            if self.args.multi_task:
                # use same subtask assignments for prediction and target
                mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
                targ_mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
            return mix_ins, targ_mix_ins

    def _make_meta_batch(self, batch: EpisodeBatch):
        reward = batch['reward']
        terminated = batch['terminated'].float()
        reset = batch['reset'].float()
        mask = batch['filled'].float()
        allocs = 1 - batch['entity2task_mask'][:, :, :self.args.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        bs, ts, _ = mask.shape
        t_added = batch['t_added'].reshape(bs, 1, 1).repeat(1, ts, 1)

        timeout = reset - terminated

        decision_points = batch['hier_decision'].float()

        seg_rewards = th.zeros_like(reward)
        cuml_rewards = th.zeros_like(reward[:, 0])
        seg_terminated = th.zeros_like(terminated)
        cuml_terminated = th.zeros_like(terminated[:, 0])
        cuml_timeout = th.zeros_like(timeout[:, 0])

        for t in reversed(range(reward.shape[1])):
            # sum rewards between hierarchical decision points
            cuml_rewards += reward[:, t]
            seg_rewards[:, t] = cuml_rewards
            cuml_rewards *= 1 - decision_points[:, t]

            # track whether env terminated between decision points
            cuml_terminated = cuml_terminated.max(terminated[:, t])
            seg_terminated[:, t] = cuml_terminated
            cuml_terminated *= 1 - decision_points[:, t]

            # mask out decision point if a env timeout happens (since we can't bootstrap from next decision point)
            cuml_timeout = cuml_timeout.max(timeout[:, t])
            mask[:, t] *= (1 - cuml_timeout)
            cuml_timeout *= 1 - decision_points[:, t]

        # scale by action length to keep gradients around same magnitude as low-level controllers
        seg_rewards /= self.args.hier_agent['action_length']

        last_alloc = th.zeros_like(allocs)
        was_reset = th.zeros_like(reset[:, [0]])
        for t in range(1, reward.shape[1]):
            # make sure that last_alloc doesn't copy final assignment from previous episode
            was_reset = (was_reset + reset[:, [t - 1]]).min(th.ones_like(was_reset))
            last_alloc[:, t] = allocs[:, t - 1] * (1 - was_reset)
            was_reset *= (1 - decision_points[:, [t]])

        # mask out last decision point in each trajectory if not terminal state (since we can't bootstrap)
        bs, ts, _ = decision_points.shape
        last_dp_ind = (
            decision_points * th.arange(
                ts, dtype=decision_points.dtype,
                device=decision_points.device).reshape(1, ts, 1)
        ).squeeze().argmax(dim=1)
        mask[th.arange(bs), last_dp_ind] *= seg_terminated[th.arange(bs), last_dp_ind]

        entity2task_mask = batch['entity2task_mask'].clone()

        d_inds = (decision_points == 1).reshape(bs, ts)
        max_bs = self.args.hier_agent['max_bs']
        meta_batch = {
            'reward': seg_rewards[d_inds][:max_bs],
            'terminated': seg_terminated[d_inds][:max_bs],
            'mask': mask[d_inds][:max_bs],
            'entities': batch['entities'][d_inds][:max_bs],
            'obs_mask': batch['obs_mask'][d_inds][:max_bs],
            'entity_mask': batch['entity_mask'][d_inds][:max_bs],
            'entity2task_mask': entity2task_mask[d_inds][:max_bs],
            'task_mask': batch['task_mask'][d_inds][:max_bs],
            'avail_actions': batch['avail_actions'][d_inds][:max_bs],
            'last_alloc': last_alloc[d_inds][:max_bs],
            't_added': t_added[d_inds][:max_bs],
        }
        return meta_batch

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        meta_batch = self._make_meta_batch(batch)      # 将原始 batch 整理成“高层分配”用的 meta batch（含 task/entity 掩码等）
        rewards = meta_batch['reward']                 # 形如 [T, B, 1] 或 [T, B] 的高层奖励序列
        terminated = meta_batch['terminated']          # 形如 [T, B, 1] 的终止标志（1 表示该时间步后 episode 结束）
        mask = meta_batch['mask']                      # 形如 [T, B, 1] 的有效步掩码（填充步为 0）
        stats = {}                                     # 用于收集训练与指标日志的字典

        # 计算当前分配（entity2task_mask 已在 batch 中存储）下的 Q 值；critic 前向，评估“已执行的任务分配”的好坏
        alloc_q, q_stats = self.mac.evaluate_allocation(meta_batch, calc_stats=True)

        # 计算新的任务分配提案（proposal）；test_mode=True 让 critic 去随机性；传入 target_mac 用于更稳定的 bootstrap 目标
        new_alloc, pi_stats = self.mac.compute_allocation(meta_batch, calc_stats=True, test_mode=True, target_mac=self.target_mac)

        # 目标 Q 值：来自策略网络在 target 上的“最佳提案”的 Q（已在 pi_stats 中算好）
        target_alloc_q = pi_stats['targ_best_prop_values']          # 形如 [T, B, 1]
        target_alloc_q = self.target_mac.alloc_critic.denormalize(target_alloc_q)  # 若 critic 做过 PopArt 归一化，这里还原到原尺度

        # TD 目标：r_t + gamma * (1 - done_t) * Q_target_{t+1}
        # 注意：若上一步是 terminal，则不 bootstrap
        targets = (rewards[:-1] + self.args.gamma * (1 - terminated[:-1]) * target_alloc_q[1:]).detach()
        if self.args.popart:
            # 若使用 PopArt，对目标做自适应标准化并更新统计（mask[:-1] 指定哪些时间步有效）
            targets = self.mac.alloc_critic.popart_update(
                targets, mask[:-1])

        td_error = (alloc_q[:-1] - targets.detach())               # TD 残差：当前 Q 与目标的差
        td_mask = mask[:-1].expand_as(td_error)                    # 对齐形状的 mask（只在有效步上计算 loss）
        if self.args.hier_agent['decay_old'] > 0:
            # 对“旧数据”衰减权重：t_added 记录该样本进入 buffer 的时间，用 cutoff 线性衰减旧样本的权重
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'][:-1].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))                # 最小不小于 0
            td_mask *= ratio                                       # 衰减应用在 mask 上 => 老样本的梯度贡献变小
        masked_td_error = td_error * td_mask                       # 仅保留有效/未被衰减到 0 的时间步
        # L2 TD-loss（MSE），对有效步取均值
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()
        stats['losses/alloc_q_loss'] = td_loss.cpu().item()        # 记录高层分配 Q 的 TD 损失

        # 反传高层 Q 的损失
        q_loss = td_loss
        self.alloc_q_optimiser.zero_grad()
        q_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)  # 梯度裁剪
        stats['train_metrics/alloc_q_grad_norm'] = grad_norm        # 记录 Q-critic 的梯度范数
        self.alloc_q_optimiser.step()

        # 分配指标：记录“最佳提案概率”（policy 对最优提案的置信度）
        stats['alloc_metrics/best_prob'] = pi_stats['best_prob'].mean().cpu().item()

        # 统计“有多少比例的 agent 改变了任务分配”（依赖上一步的分配是否存在）
        active_ag = 1 - meta_batch['entity_mask'][:, :self.args.n_agents].float()  # 有效 agent 掩码（1=有效），shape [T, B, n_agents]
        ag_changed = (meta_batch['last_alloc'].argmax(dim=2) != new_alloc.detach().argmax(dim=2)).float()  # 每个 agent 是否更换任务
        prev_al_exists = (meta_batch['last_alloc'].sum(dim=(1, 2)) >= 1).float()   # 之前是否存在分配（避免第一步无前态）
        perc_changed_per_step = ((ag_changed * active_ag).sum(dim=1) / active_ag.sum(dim=1))  # 每步有效 agent 的变化比例
        perc_changed = (perc_changed_per_step * prev_al_exists).sum() / prev_al_exists.sum()  # 只在“有前态”的步上取平均
        stats['alloc_metrics/perc_ag_changed'] = perc_changed.cpu().item()

        # 计算每个子任务上“agent 数量 vs 非 agent 实体数量”的绝对差（仅在某些任务有意义）
        nonagent2task = 1 - meta_batch['entity2task_mask'][:, self.args.n_agents:].float()  # 非 agent 的实体-任务掩码
        ag_per_task = new_alloc.detach().sum(dim=1)                  # 每个任务分配到的 agent 数，shape [T, B, n_tasks]
        nag_per_task = nonagent2task.sum(dim=1)                      # 每个任务对应的“非 agent 实体（如建筑等）数量”
        absdiff_per_task = (ag_per_task - nag_per_task).abs()        # 两者的绝对差
        abs_diff_mean = absdiff_per_task.sum(dim=1) / (1 - meta_batch['task_mask'].float()).sum(dim=1)  # 对有效任务平均
        stats['alloc_metrics/ag_task_concentration'] = abs_diff_mean.mean().cpu().item()  # 任务“集中度”指标（越大可能越不均衡）

        # AQL 的策略摊销（amortization）目标：最大化“最佳提案”的对数概率
        all_prop_log_pi = pi_stats['log_pi']                         # 所有采样提案的 log π，shape [T, B, n_prop]
        bs = all_prop_log_pi.shape[0]
        best_prop_log_pi = all_prop_log_pi[th.arange(bs), pi_stats['best_prop_inds']]  # 每步选出“最佳”提案对应的 log π
        amort_step_loss = -best_prop_log_pi                          # 负对数似然（最大化 log π <=> 最小化 -log π）
        masked_amort_step_loss = amort_step_loss * mask              # 只在有效步上计算
        amort_loss = masked_amort_step_loss.sum() / mask.sum()       # 对有效步做归一化平均
        stats['losses/alloc_amort_loss'] = amort_loss.cpu().item()   # 记录摊销损失（把 critic 偏好蒸馏到 policy）

        # 统计是否存在“有任务但没有任何 agent 被分配”的情况（用于诊断分配质量）
        active_task = 1 - meta_batch['task_mask'].float().unsqueeze(1)  # 有效任务掩码，shape [T, 1, n_tasks]
        ag2task = pi_stats['all_allocs'].detach()  # 所有提案的 agent→task 分配（多样本），shape [T, n_prop, n_agents, n_tasks]
        task_has_agents = (ag2task.sum(dim=2) > 0).float()              # 每个提案下，每个任务是否至少有一个 agent
        any_task_no_agents = (task_has_agents.sum(dim=2, keepdim=True)
                              != active_task.sum(dim=2, keepdim=True)).float()  # 若激活任务数 != 有 agent 的任务数，则说明有空缺
        stats['alloc_metrics/any_task_no_agents_pi'] = any_task_no_agents.mean().cpu().item()

        # 策略熵（鼓励探索/分配多样性）：这里取负作为正则（越“确定”惩罚越小，越“随机”惩罚越大或越小取决于权重）
        entropy = pi_stats['entropy']                                  # 形如 [T, B] 或 [T, B, 1]
        entropy_loss = -entropy.mean()                                  # 负熵作为损失项
        stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()      # 记录熵本身（正值），便于观测

        # 最终 policy loss = 摊销损失 + 熵正则（带权重）
        pi_loss = (amort_loss
                   + self.args.hier_agent['entropy_loss'] * entropy_loss)

        # 反传策略损失（仅更新分配 policy 的参数）
        self.alloc_pi_optimiser.zero_grad()
        pi_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)  # 梯度裁剪
        stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
        self.alloc_pi_optimiser.step()

        # 按间隔更新“高层分配的 target 网络”（提高训练稳定性）
        if (episode_num - self.last_alloc_target_update_episode) / self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num

        # 定期写入日志（按 t_env 的间隔）
        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env

        return stats, new_alloc                                       # 返回本次训练的指标与新分配（供上层使用/可视化）

    def _broadcast_decisions_to_batch(self, decisions, decision_pts):
        decision_pts = decision_pts.squeeze(-1)
        bs, ts = decision_pts.shape
        bcast_decisions = {k: th.zeros_like(v[[0]]).unsqueeze(0).repeat(bs * rep, ts, *(1 for _ in range(len(v.shape) - 1))) for k, (v, rep) in decisions.items()}
        for decname in bcast_decisions:
            value, rep = decisions[decname]
            bcast_decisions[decname][decision_pts.repeat(rep, 1)] = value
        for t in range(1, ts):
            for decname in bcast_decisions:
                rep = decisions[decname][1]
                prev_value = bcast_decisions[decname][:, t - 1]
                bcast_decisions[decname][:, t] = ((decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1).to(prev_value.dtype) * bcast_decisions[decname][:, t])
                                                  + ((1 - decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1)).to(prev_value.dtype) * prev_value))
        return bcast_decisions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 取出需要的张量（默认：时间维 T，批维 B，可能还带 agent/action 维）
        rewards = batch["reward"][:, :-1]                   # [B, T-1, 1] 或 [B, T-1]：环境奖励（去掉最后一步—无下个状态）
        actions = batch["actions"][:, :-1]                  # [B, T-1, n_agents, 1]：离散动作索引（one-hot 索引形式）

        # episode 结束标志（不含 timeout）：用于决定是否 bootstrap
        terminated = batch["terminated"][:, :-1].float()    # [B, T-1, 1]，1 表示该步后自然终止（非超时）

        # reset：episode 在该时间步之后重置（自然结束或超时），用于决定哪些时间步可用于学习
        # 最后一个时间步因无转移（无 next state）不能学习
        reset = batch["reset"][:, :-1].float()              # [B, T-1, 1]

        mask = batch["filled"][:, :-1].float()              # [B, T-1, 1]：有效步掩码（padding 为 0）
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])     # 对于 reset 触发后的步（下一步），不学习（去掉那些转移）
        org_mask = mask.clone()                             # 备份原始 mask（COPA VI loss 中会用）

        avail_actions = batch["avail_actions"]              # [B, T, n_agents, n_actions]：可执行动作掩码

        if self.args.agent['subtask_cond'] is not None:
            # 子任务条件训练：为每个 task 学一套控制器（多头/多任务学习）
            rewards = batch['task_rewards'][:, :-1]         # [B, T-1, n_tasks]：按任务分解的奖励
            terminated = batch['tasks_terminated'][:, :-1].float()  # [B, T-1, n_tasks]：每个任务的终止
            mask = mask.repeat(1, 1, self.args.n_tasks)     # [B, T-1, n_tasks]：扩展 mask 到每个任务
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])    # 任务级的 reset
            # 仅在“该任务有 agent 被分配”的时间步学习
            task_has_agents = (1 - batch['entity2task_mask'][:, :-1, :self.args.n_agents]).sum(2) > 0  # [B, T-1]
            mask *= task_has_agents.float().unsqueeze(-1)   # 无 agent 的任务置 0（不可学习）

        # 初始化 RNN 隐状态
        self.mac.init_hidden(batch.batch_size)

        # 训练/评估模式设置：mac 与 mixer 启用训练（如 dropout），target 版本保持 eval（固定）
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        # 以下变量用于 COPA（层次教练）模块：coach_h/coach_z 是高层策略的表征/潜变量
        coach_h = None
        targ_coach_h = None
        coach_z = None
        targ_coach_z = None

        imagine_inps = None
        if self.args.agent['imagine']:
            # REFIL/Imagine：构造“想象”输入（augmented inputs），在同一 batch 上多视角训练
            imagine_inps, imagine_groups = self.mac.agent.make_imagined_inputs(batch)

        if self.use_copa:
            # 编码器抽取高层观测表征（coach_h），target 分支同理
            coach_h = self.mac.coach.encode(batch, imagine_inps=imagine_inps)  # [B*T, ...]（实现相关）
            targ_coach_h = self.target_mac.coach.encode(batch)

            decision_points = batch['hier_decision'].squeeze(-1)  # [B*T]：哪些时间步做高层决策
            bs_rep = 1
            if self.args.agent['imagine']:
                bs_rep = 3                                       # REFIL 可能复制 batch（真实+两种想象）=> 3 倍

            # 取出决策时刻的高层表征（真实与 target），用于采样/计算潜变量 z
            coach_h_t0 = coach_h[decision_points.repeat(bs_rep, 1)]
            targ_coach_h_t0 = targ_coach_h[decision_points]

            # 从策略网络得到潜变量 z 及其（均值/方差）参数（用于 VI）
            coach_z_t0, coach_mu_t0, coach_logvar_t0 = self.mac.coach.strategy(coach_h_t0)
            coach_mu_t0 = coach_mu_t0.chunk(bs_rep, dim=0)[0]          # 仅取真实分支的参数参与 VI
            coach_logvar_t0 = coach_logvar_t0.chunk(bs_rep, dim=0)[0]
            targ_coach_z_t0, _, _ = self.target_mac.coach.strategy(targ_coach_h_t0)

            # 把决策时刻的变量广播回完整时间轴（非决策步可复用最近一次决策的 z）
            bcast_ins = {
                'coach_z_t0': (coach_z_t0, bs_rep),
                'coach_mu_t0': (coach_mu_t0, 1),
                'coach_logvar_t0': (coach_logvar_t0, 1),
                'targ_coach_z_t0': (targ_coach_z_t0, 1),
            }
            bcast_decisions = self._broadcast_decisions_to_batch(bcast_ins, batch['hier_decision'])
            coach_z = bcast_decisions['coach_z_t0']                   # [B, T, ...]：每步的高层潜变量输入
            coach_mu = bcast_decisions['coach_mu_t0']                 # 仅用于 VI
            coach_logvar = bcast_decisions['coach_logvar_t0']
            targ_coach_z = bcast_decisions['targ_coach_z_t0']        # target 分支的潜变量（仅用于目标）

        # 若启用 imagine，会把同一 batch 复制（真实 + 2 份想象），在 0 维堆叠
        batch_mult = 1
        if self.args.agent['imagine']:
            batch_mult += 2                                          # 共 3 份：真实+两种变换

        # 前向：得到所有时间步的每个 agent 的 Q(a|s)（叠了 batch_mult）
        all_mac_out, mac_info = self.mac.forward(
            batch, t=None,
            coach_z=coach_z,
            imagine_inps=imagine_inps)                                # [batch_mult*B, T, n_agents, n_actions]

        # 将动作索引扩展成与 all_mac_out 对齐的重复批
        rep_actions = actions.repeat(batch_mult, 1, 1, 1)             # [batch_mult*B, T-1, n_agents, 1]

        # 选取“被执行动作”的 Q 值 Q(s,a)（去掉最后一时刻）
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # [batch_mult*B, T-1, n_agents]

        # 将 concat 的 batch 拆回（真实 vs 想象）
        mac_out_tup = all_mac_out.chunk(batch_mult, dim=0)
        caq_tup = all_chosen_action_qvals.chunk(batch_mult, dim=0)

        mac_out = mac_out_tup[0]                                      # 真实分支的 Q 分布 [B, T, n_agents, n_actions]
        chosen_action_qvals = caq_tup[0]                               # 真实分支的 Q(s,a) [B, T-1, n_agents]
        if self.args.agent['imagine']:
            # 想象分支的 Q(s,a) 合并在时间维后面以便共同过 mixer
            caq_imagine = th.cat(caq_tup[1:], dim=2)                   # [B, T-1, n_agents*2]（两个 imagine 叠到 agent 维）

        # target 网络也要初始化 RNN 隐状态
        self.target_mac.init_hidden(batch.batch_size)

        # target 前向：得到 target Q 分布
        target_mac_out, _ = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)  # [B, T, n_agents, n_actions]

        # 处理可用动作掩码（target 使用的是下一个时刻的可用动作）
        if self.args.agent['subtask_cond'] is not None:
            # 若做子任务条件控制，需要把 avail_actions 按任务分配掩码裁剪
            allocs = (1 - batch['entity2task_mask'][:, :, :self.args.n_agents])       # [B, T, n_agents, n_tasks]
            avail_actions_targ = parse_avail_actions(avail_actions[:, 1:], allocs[:, :-1], self.args)
        else:
            avail_actions_targ = avail_actions[:, 1:]                                  # [B, T-1, n_agents, n_actions]

        target_mac_out = target_mac_out[:, 1:]                                         # 与 target 对齐成 T-1

        # 将 target 中不可用动作置极小，防止被 max 选到
        target_mac_out[avail_actions_targ == 0] = -9999999

        # 计算 bootstrap 目标 Q_max(s', ·)：
        if self.args.double_q:
            # Double Q-Learning：用当前网络选动作，用 target 网络取该动作的 Q
            mac_out_detach = mac_out.clone().detach()[:, 1:]           # 当前网络的 Q（下一个时刻）
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]   # argmax_a Q(s',a)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)  # Q_target(s', argmax_a Q)
        else:
            # 普通 DQN：直接在 target 上取 max
            target_max_qvals = target_mac_out.max(dim=3)[0]                 # [B, T-1, n_agents]

        # 混合器（VDN/QMIX/FlexQMIX）：将各 agent Q 聚合成全局 Q_tot
        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)              # mixer 的状态/实体级输入

            chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)   # Q_tot(s, u) [B, T-1, 1]
            gamma = self.args.gamma

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)  # Q_tot^target(s', u*)
            target_max_qvals = self.target_mixer.denormalize(target_max_qvals)    # 若 target_mixer 做过 PopArt，需反归一化

            # 1-step Bellman 目标：r + γ * (1 - done) * Q_tot^target(s', u*)
            targets = (rewards + gamma * (1 - terminated) * target_max_qvals).detach()  # [B, T-1, 1]

            if self.args.popart:
                # 用当前 mixer 的 PopArt 对 target 做自适应标准化，并更新统计
                targets = self.mixer.popart_update(
                    targets, mask)

            if self.args.agent['imagine']:
                # 想象分支：也要过 mixer（但不包含最后一个时间步）
                imagine_groups = [gr[:, :-1] for gr in imagine_groups]        # 对齐 T-1
                caq_imagine = self.mixer(caq_imagine, mix_ins,
                                         imagine_groups=imagine_groups)       # [B, T-1, 1]
        else:
            # 无 mixer（纯 per-agent 学习的情况很少见）：用逐 agent 的 max 直接构造目标
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        # TD 误差（真实分支）
        td_error = (chosen_action_qvals - targets.detach())           # [B, T-1, 1] 或 [B, T-1, n_agents]（取决于是否 mixer）
        mask = mask.expand_as(td_error)                               # 广播 mask 到相同 shape
        masked_td_error = td_error * mask

        # L2 TD 损失（MSE），对有效元素做归一化平均
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.agent['imagine']:
            # REFIL：真实与想象分支的 TD loss 按 λ 加权融合
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())            # 想象分支的 TD 残差
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss           # 总损失 = (1-λ)*真实 + λ*想象

        if self.use_copa and self.args.hier_agent['copa_vi_loss']:
            # COPA 的变分推断（VI）损失：让识别器 q(z|·) 拟合策略先验 p(z|·)，并加上先验熵（鼓励信息量）
            q_mu, q_logvar = self.mac.copa_recog(batch)               # 识别器给出 q(z|·) 的参数
            q_t = D.normal.Normal(q_mu, (0.5 * q_logvar).exp())
            coach_z = coach_z.chunk(bs_rep, dim=0)[0]                 # 只对真实分支训练 VI
            log_prob = q_t.log_prob(coach_z).clamp_(-1000, 0).sum(-1) # log q(z|·)（裁剪数值，防 NaN）

            # 先验 p(z|·) 的熵，鼓励分布有一定宽度（避免塌缩）
            p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
            entropy = p_.entropy().clamp_(0, 10).sum(-1)

            # 屏蔽无效 agent（有些实体不是 agent）
            agent_mask = 1 - batch['entity_mask'][:, :, :self.args.n_agents].float()  # [B, T, n_agents]
            log_prob = (log_prob * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)  # 平均到 agent 维
            entropy = (entropy * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)

            # 时间维上按 org_mask 取有效元素（忽略末步/重置步）
            vi_loss = (-log_prob[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            entropy_loss = (-entropy[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()

            # 将 VI 项并入总损失（权重 vi_lambda；熵项的权重再 /10）
            loss += vi_loss * self.args.vi_lambda + entropy_loss * self.args.vi_lambda / 10

        # 优化器步骤：主网络（agent、mixer、以及可能的 coach 等）
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # 梯度裁剪
        self.optimiser.step()

        # 按间隔更新主 Q 的 target 网络（提高训练稳定性）
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 定期记录训练日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("losses/q_loss", loss.item(), t_env)
            if self.args.agent['imagine']:
                self.logger.log_stat("losses/im_loss", im_loss.item(), t_env)
            if self.use_copa and self.args.hier_agent['copa_vi_loss']:
                self.logger.log_stat("losses/copa_vi_loss", vi_loss.item(), t_env)
                self.logger.log_stat("losses/copa_entropy_loss", entropy_loss.item(), t_env)

            self.logger.log_stat("train_metrics/q_grad_norm", grad_norm, t_env)

            mask_elems = mask.sum().item()
            # 平均绝对 TD 误差（仅在有效元素上）
            self.logger.log_stat("train_metrics/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # 选中动作 Q 的均值（按 agent 与 mask 归一）
            self.logger.log_stat("train_metrics/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # 目标值（targets）的均值（按 agent 与 mask 归一）
            self.logger.log_stat("train_metrics/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_alloc_targets(self):
        self.target_mac.load_alloc_state(self.mac)
        self.logger.console_logger.info("Updated allocation target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}opt.th".format(path))

    def load_models(self, path, pi_only=False, evaluate=False):
        self.mac.load_models(path, pi_only=pi_only)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path, pi_only=pi_only)
        if not evaluate and not pi_only:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}opt.th".format(path), map_location=lambda storage, loc: storage))
