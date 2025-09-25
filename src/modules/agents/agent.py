import torch as th
import torch.nn as nn
from .bases import EntityBase, StandardBase
from .heads import RecurrentHead, FeedforwardHead
from .allocation_common import TaskEmbedder

'''
Base（EntityBase/StandardBase）：将输入状态编码成每个 agent 的隐藏表示（支持实体注意力、掩码可见性等）。
Head（Recurrent/Feedforward）：把 Base 的表示变成每个 agent 对每个动作的 Q 值；若是 RNN，会维护 h。
CoPA（可选）：若启用，会把层级策略的潜变量 coach_z 加到编码表示中，注入高层意图。
subtask_cond：
    'mask'：把“同任务的实体”可见，跨任务不可见（通过 obs_mask 控制注意力）。
    'full_obs'：为实体拼上任务嵌入向量（task_embeds），让网络显式“看见”任务归属。
REFIL 想象输入：make_imagined_inputs 会复制 batch 两份，构造“组内交互/组间交互”的两种注意力场景，以提升泛化和鲁棒性（训练时与真实 batch 拼接在一起共同前向）。
掩码语义统一：这里的各种 mask 都使用 0=允许/可见、1=禁止/不可见 的约定，保证与注意力/可观测性逻辑一致。
无效代理清零：通过 entity_mask 把不存在的 agent 的 Q 置零，防止干扰损失。
'''

class Agent(nn.Module):
    def __init__(self, input_shape, args,
                 recurrent=False, entity_scheme=True,
                 subtask_cond=None,
                 **kwargs):
        super().__init__()                                      # 初始化 nn.Module
        self.args = args
        if subtask_cond == 'full_obs':
            self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
            # 若子任务条件为 full_obs，准备一个将“实体×任务分配矩阵”映射到任务嵌入的层

        if entity_scheme:
            self._base = EntityBase(input_shape, args)          # 实体方案：图/注意力式编码器
        else:
            self._base = StandardBase(input_shape, args)        # 标准方案：非实体式编码器（平铺特征）

        out_dim = args.n_actions                                # 动作空间维度（Q 值输出的最后一维）

        self.use_copa = self.args.hier_agent['copa']            # 是否启用 CoPA（层级策略潜变量）

        head_in_dim = args.rnn_hidden_dim                       # 进入 head 的隐藏维（由 base 决定输出维）
        if recurrent:
            self._head = RecurrentHead(args, in_dim=head_in_dim, out_dim=out_dim)
            # 循环式 head：给每个 agent 产生 Q 值（且维护 RNN 隐状态）
        else:
            self._head = FeedforwardHead(args, in_dim=head_in_dim, out_dim=out_dim)
            # 前馈式 head：不带时序状态

        self.subtask_cond = subtask_cond                        # 子任务条件的模式（None/'mask'/'full_obs'）

    def init_hidden(self):
        # 让隐藏状态落在与模型相同的 device 上
        return self._base.init_hidden()

    def _compute_network(self, inputs):
        x = self._base(inputs)                                  # 先用 Base 编码实体/状态
        if self.use_copa:
            x += inputs['coach_z']                              # 若用 CoPA，将层级潜变量 z 融合到表示中（残差相加）
        q, h = self._head(x, inputs)                            # 通过 Head 计算每个 agent 的 Q 值，以及新的隐状态 h
        if 'entity_mask' in inputs:
            entity_mask = inputs['entity_mask']                 # (bs, ts, n_entities)，1 表示“该实体无效/不存在”
            agent_mask = entity_mask[:, :, :self.args.n_agents] # 取前 n_agents 列（仅代理）
            # 对无效 agent 的输出 Q 做 0 遮罩，避免污染 loss/梯度
            q = q.masked_fill(agent_mask.unsqueeze(3).bool(), 0)
        return q, h

    def _logical_not(self, inp):
        return 1 - inp                                          # 取反（输入是 0/1 ByteTensor）

    def _logical_or(self, inp1, inp2):
        out = inp1 + inp2                                       # 逻辑或：逐元素相加后截断到 1
        out[out > 1] = 1
        return out

    def _groupmask2attnmask(self, group_mask):
        """
        将“实体属于哪个组”的 one-hot / multi-hot 掩码转换为注意力矩阵掩码：
        只允许同组实体之间相互注意（attend）。
        输入：
          - group_mask: (bs, ts, ne[, ng])，最后一维是组维度，可省略（单组）
        输出：
          - attn_mask: (bs, ts, ne, ne)，0=可见/可注意, 1=不可注意
        """
        if len(group_mask.shape) == 3:
            bs, ts, ne = group_mask.shape
            ng = 1
        elif len(group_mask.shape) == 4:
            bs, ts, ne, ng = group_mask.shape
        else:
            raise Exception("Unrecognized group_mask shape")

        in1 = (1 - group_mask.to(th.float)).reshape(bs * ts, ne, ng)  # 1 表示“不在该组”
        in2 = in1.transpose(1, 2)                                     # (bs*ts, ng, ne)
        attn_mask = 1 - th.bmm(in1, in2)                              # (bs*ts, ne, ne)，同组得到 0，可相互注意
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)         # 转回 (bs, ts, ne, ne)，Byte 掩码

    def make_imagined_inputs(self, inputs):
        """
        生成 REFIL/想象训练所需的“伪输入”：
        - 随机把一集里的实体划分为 A/B 两组，构造“组内交互”和“组间交互”两种注意力视角
        - 返回复制的输入（batch 在维度 0 上拼接 2 倍），以及供 Mixer 用的两种注意力掩码
        """
        entities = inputs['entities']                       # (bs, ts, ne, ed) 实体特征
        obs_mask = inputs['obs_mask']                       # (bs, ts, ne, ne) 观测可见性掩码（1=不可见）
        entity_mask = inputs['entity_mask']                 # (bs, ts, ne)     实体是否存在（1=不存在）
        bs, ts, ne, ed = entities.shape

        # 每条 episode 采样一次将实体随机分成 A/B 两组的概率（所有实体同一概率 p）
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)    # (bs, 1, ne) A 组二值
        groupB = self._logical_not(groupA)                  # B 组为 A 的补集
        # 对于环境中“无效/不存在”的实体，强制标 1（即：不参与注意力）
        groupA = self._logical_or(groupA, entity_mask[:, [0]])
        groupB = self._logical_or(groupB, entity_mask[:, [0]])

        # 将组掩码变为“只能同组注意”的注意力掩码（0=允许，1=禁止）
        groupAattnmask = self._groupmask2attnmask(groupA)   # (bs, 1, ne, ne)
        groupBattnmask = self._groupmask2attnmask(groupB)

        # 组间交互：允许跨组注意（= 非(各自组内掩码) 的或）
        interactattnmask = self._logical_or(self._logical_not(groupAattnmask),
                                            self._logical_not(groupBattnmask))
        # 组内交互：与组间互为补关系
        withinattnmask = self._logical_not(interactattnmask)

        # 仅允许“存在的实体”彼此注意，用 entity_mask 生成一个“基础不可见”掩码
        activeattnmask = self._groupmask2attnmask(entity_mask[:, [0]])

        # Mixer 用的版本（不叠加 obs_mask，可反映“结构性”想象）
        Wattnmask_noobs = self._logical_or(withinattnmask, activeattnmask).repeat(1, ts, 1, 1)
        Iattnmask_noobs = self._logical_or(interactattnmask, activeattnmask).repeat(1, ts, 1, 1)

        # 模型前向用的版本：还要叠加观测可见性 obs_mask（同一语义：1=不可注意）
        withinattnmask = self._logical_or(withinattnmask, obs_mask)
        interactattnmask = self._logical_or(interactattnmask, obs_mask)

        new_inputs = {}
        if 'entity2task_mask' in inputs:
            # 任务分配掩码也复制两份，与想象视角对齐（batch 维拼接）
            new_inputs['entity2task_mask'] = inputs['entity2task_mask'].repeat(2, 1, 1, 1)

        # 把实体特征/掩码等统一复制两份：第一份用于“组内视角”，第二份用于“组间视角”
        new_inputs['entities'] = entities.repeat(2, 1, 1, 1)
        new_inputs['obs_mask'] = th.cat([withinattnmask, interactattnmask], dim=0)  # 拼 batch 维
        new_inputs['imagine_mask'] = th.cat([Wattnmask_noobs, Iattnmask_noobs], dim=0)
        new_inputs['entity_mask'] = entity_mask.repeat(2, 1, 1)
        new_inputs['reset'] = inputs['reset'].repeat(2, 1, 1)

        # 返回：拼好的想象输入，以及给 Mixer 用的 (组内, 组间) 掩码对
        return new_inputs, (Wattnmask_noobs, Iattnmask_noobs)

    def _mask_by_task(self, inputs):
        """
        若选择了 'mask' 子任务条件：
        将 entity2task_mask（实体×任务）转为“同任务实体互相可见、跨任务不可见”的注意力掩码，
        并 merge 到 inputs['obs_mask'] 中。
        """
        entity2task_mask = inputs['entity2task_mask']   # (bs, ts, ne, nt)，[i,j]=0 表示实体 i 属于任务 j
        task_obs_mask = self._groupmask2attnmask(entity2task_mask)
        inputs['obs_mask'] = self._logical_or(inputs['obs_mask'], task_obs_mask)
        return inputs

    def _observe_tasks(self, inputs):
        """
        若选择了 'full_obs' 子任务条件：
        计算每个实体的任务嵌入（基于 1 - entity2task_mask 的归属 one-hot），
        并放入 inputs['task_embeds']，供 Base/Head 使用。
        """
        entity2task = 1 - inputs['entity2task_mask'].float()     # 归属 one-hot：1=属于该任务
        inputs['task_embeds'] = self.task_embed(entity2task)     # (bs, ts, ne, embed_dim)
        return inputs

    def forward(self, inputs, imagine_inps=None):
        info = {}
        input_list = [inputs]                                    # 主输入（真实观测）

        # 根据子任务条件模式，对输入做增强
        if self.subtask_cond == 'mask':
            input_list = [self._mask_by_task(inp) for inp in input_list]
        if self.subtask_cond == 'full_obs':
            input_list = [self._observe_tasks(inp) for inp in input_list]
        if self.subtask_cond is not None and self.subtask_cond not in ['mask', 'full_obs']:
            raise Exception("Subtask conditioning not recognized")

        if imagine_inps is not None:
            input_list.append(imagine_inps)                      # 若有“想象视角”输入，把它当作额外 batch 拼接

        # 只保留多个输入 dict 的公共键（保证能沿 batch 维拼接）
        in_keys = set.intersection(*[set(d) for d in input_list])

        coach_z = None
        if 'coach_z' in inputs:
            # coach_z 只在真实 inputs 中，故不会出现在 in_keys 交集中；单独处理并随后回填
            coach_z = inputs['coach_z']

        # 对于公共键，把各个输入 dict 在 batch 维（维度 0）拼在一起，形成“大 batch”
        inputs = {k: th.cat([inp_dict[k] for inp_dict in input_list], dim=0)
                  for k in in_keys}
        if coach_z is not None:
            inputs['coach_z'] = coach_z                          # 回填 coach_z（仅对真实 batch 生效）

        q, h = self._compute_network(inputs)                     # 统一送入 base + head，得到 Q 与隐状态
        return q, h, info                                        # info 预留：当前实现返回空 dict
