# ALMA 下层Q网络训练流程详解

## 概述

ALMA的下层Q网络训练采用标准的深度Q学习（DQN）方法，使用经验回放（Experience Replay）机制。整个流程包括：**轨迹收集 → 轨迹存储 → 轨迹采样 → 网络训练**。

---

## 一、轨迹收集（Data Collection）

### 1.1 主训练循环

**文件位置**：`/data/gu-di/ALMA/src/run.py` (第355-372行)

```python
while runner.t_env <= args.t_max:
    # 1. 运行一个完整episode，收集轨迹
    episode_batch, _ = runner.run(test_mode=False)
    
    # 2. 将轨迹存储到经验回放缓冲区
    buffer.insert_episode_batch(episode_batch)
    
    # 3. 如果缓冲区有足够数据，开始训练
    if buffer.can_sample(args.batch_size):
        for _ in range(args.training_iters):
            # 4. 从缓冲区采样一批轨迹
            episode_sample = buffer.sample(args.batch_size)
            
            # 5. 训练下层Q网络
            learner.train(episode_sample, runner.t_env, episode)
```

### 1.2 EpisodeRunner.run() - 轨迹收集核心

**文件位置**：`/data/gu-di/ALMA/src/runners/episode_runner.py` (第76-171行)

**关键步骤**：

```python
def run(self, test_mode=False, ...):
    # 1. 重置环境
    self.reset(test=test_scen, index=index, n_tasks=n_tasks)
    
    # 2. 初始化隐藏状态
    self.mac.init_hidden(batch_size=self.batch_size)
    
    # 3. 运行episode，每一步收集数据
    while not terminated:
        # 3.1 获取当前状态（obs, avail_actions等）
        pre_transition_data = self._get_pre_transition_data(env_info)
        self.batch.update(pre_transition_data, ts=self.t)
        
        # 3.2 使用下层Q网络选择动作
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        
        # 3.3 执行动作，获得奖励和下一状态
        reward, terminated, env_info = self.env.step(actions[0].cpu())
        
        # 3.4 存储transition数据（actions, reward, terminated, reset）
        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
            "reset": [(terminated,)],
        }
        self.batch.update(post_transition_data, ts=self.t)
        
        self.t += 1
    
    # 4. 返回完整的episode轨迹
    return self.batch, final_subtask_infos
```

**收集的数据**：
- `obs`: 智能体观察
- `state`: 全局状态（如果使用）
- `entities`: 实体信息（如果使用entity_scheme）
- `avail_actions`: 可用动作掩码
- `actions`: 选择的动作
- `reward`: 奖励
- `terminated`: 是否终止（非超时）
- `reset`: 是否重置（包括超时）

### 1.3 动作选择（使用下层Q网络）

**文件位置**：`/data/gu-di/ALMA/src/controllers/basic_controller.py`

在 `runner.run()` 中调用 `self.mac.select_actions()`，这会：
1. 调用 `mac.forward()` 计算Q值
2. 使用 `action_selector`（通常是epsilon-greedy）选择动作

---

## 二、轨迹存储（Experience Replay Buffer）

### 2.1 插入轨迹

**文件位置**：`/data/gu-di/ALMA/src/components/episode_buffer.py` (第311-326行)

```python
def insert_episode_batch(self, ep_batch):
    if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
        # 存储transition数据（obs, actions, rewards等）
        self.update(ep_batch.data.transition_data,
                    slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                    slice(0, ep_batch.max_seq_length),
                    mark_filled=False)
        # 存储episode数据（ep_num, t_added等）
        self.update(ep_batch.data.episode_data,
                    slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
        self.buffer_index = (self.buffer_index + ep_batch.batch_size) % self.buffer_size
```

**特点**：
- 使用循环缓冲区（circular buffer）
- 支持高效存储（EfficientStore）避免padding浪费内存
- 自动覆盖旧数据

### 2.2 缓冲区结构

**文件位置**：`/data/gu-di/ALMA/src/components/episode_buffer.py` (第44-468行)

- `ReplayBuffer`: 经验回放缓冲区主类
- `EpisodeBatch`: 存储单个episode的数据
- `EfficientStore`: 高效存储变长轨迹，避免padding

---

## 三、轨迹采样（Sampling）

### 3.1 采样方法

**文件位置**：`/data/gu-di/ALMA/src/components/episode_buffer.py` (第346-391行)

```python
def sample(self, batch_size, filters={}):
    # 1. 创建过滤掩码（可选）
    filtered_ep_mask = th.zeros(self.buffer_size, dtype=th.bool, device=self.device)
    filtered_ep_mask[:self.episodes_in_buffer] = 1
    for item_name, condition in filters.items():
        filtered_ep_mask *= condition(self.data.episode_data[item_name].flatten())
    
    # 2. 随机采样episode索引
    valid_inds = th.arange(self.buffer_size, device=self.device)[filtered_ep_mask]
    ep_ids = np.random.choice(valid_inds.cpu().numpy(), batch_size, replace=False)
    
    # 3. 获取采样的episode数据
    batch = self[ep_ids]
    
    # 4. 如果设置了max_traj_len，截断轨迹
    if self.max_traj_len != -1:
        # 随机选择连续的子轨迹
        # ... 截断逻辑 ...
    
    return batch
```

**特点**：
- 均匀随机采样
- 支持过滤条件（如只采样最近的轨迹）
- 支持轨迹截断（max_traj_len）

---

## 四、网络训练（Training）

### 4.1 训练主函数

**文件位置**：`/data/gu-di/ALMA/src/learners/q_learner.py` (第293-515行)

```python
def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    # 1. 提取数据
    rewards = batch["reward"][:, :-1]  # [bs, ts-1, 1]
    actions = batch["actions"][:, :-1]  # [bs, ts-1, na, 1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    avail_actions = batch["avail_actions"]
    
    # 2. 计算当前Q值（使用主网络）
    self.mac.init_hidden(batch.batch_size)
    self.mac.train()  # 启用训练模式（dropout等）
    
    # 前向传播，计算所有动作的Q值
    all_mac_out, mac_info = self.mac.forward(batch, t=None, coach_z=coach_z)
    # all_mac_out: [bs, ts, na, n_actions]
    
    # 提取选择的动作对应的Q值
    chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=actions).squeeze(3)
    # chosen_action_qvals: [bs, ts-1, na]
    
    # 3. 计算目标Q值（使用目标网络）
    self.target_mac.init_hidden(batch.batch_size)
    self.target_mac.eval()  # 目标网络保持评估模式
    
    target_mac_out, _ = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)
    target_mac_out = target_mac_out[:, 1:]  # [bs, ts-1, na, n_actions]
    
    # 掩码不可用动作
    target_mac_out[avail_actions_targ == 0] = -9999999
    
    # 选择最大Q值（或使用double Q-learning）
    if self.args.double_q:
        # Double Q-learning: 使用主网络选择动作，目标网络评估
        mac_out_detach = mac_out.clone().detach()[:, 1:]
        mac_out_detach[avail_actions_targ == 0] = -9999999
        cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
    else:
        # 标准Q-learning: 直接取最大值
        target_max_qvals = target_mac_out.max(dim=3)[0]
    
    # 4. 使用Mixer混合Q值（如果使用）
    if self.mixer is not None:
        chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
        target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
        target_max_qvals = self.target_mixer.denormalize(target_max_qvals)
    
    # 5. 计算TD目标
    targets = (rewards + gamma * (1 - terminated) * target_max_qvals).detach()
    
    # 6. 计算TD误差和损失
    td_error = (chosen_action_qvals - targets.detach())
    mask = mask.expand_as(td_error)
    masked_td_error = td_error * mask
    loss = (masked_td_error ** 2).sum() / mask.sum()
    
    # 7. 反向传播和优化
    self.optimiser.zero_grad()
    loss.backward()
    grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
    self.optimiser.step()
    
    # 8. 定期更新目标网络
    if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
        self._update_targets()
        self.last_target_update_episode = episode_num
```

### 4.2 MAC前向传播（下层Q网络）

**文件位置**：`/data/gu-di/ALMA/src/controllers/basic_controller.py` (第64-85行)

```python
def forward(self, ep_batch, t, coach_z=None, ...):
    # 1. 构建输入（obs, last_action, agent_id等）
    agent_inputs, imagine_inps = self._build_inputs(ep_batch, t, target=target)
    agent_inputs['hidden_state'] = self.hidden_states
    
    # 2. 如果使用COPA，添加coach_z
    if self.use_copa:
        agent_inputs['coach_z'] = coach_z
    
    # 3. 通过Agent网络（下层Q网络）前向传播
    agent_outs, self.hidden_states, info = self.agent(agent_inputs, imagine_inps=imagine_inps)
    # agent_outs: [bs, ts, na, n_actions] - 每个智能体每个动作的Q值
    
    return agent_outs, info
```

**Agent网络结构**（通常在`modules/agents/agent.py`中）：
- 输入：观察（obs）+ 隐藏状态（hidden_state）
- 处理：RNN/GRU处理序列信息
- 输出：每个动作的Q值 [bs, ts, na, n_actions]

---

## 五、完整训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    主训练循环 (run.py)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. 轨迹收集 (EpisodeRunner.run())    │
        │     - 运行一个episode                  │
        │     - 每一步收集 (obs, action, reward) │
        │     - 使用下层Q网络选择动作             │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  2. 轨迹存储 (buffer.insert_...)       │
        │     - 存储到经验回放缓冲区             │
        │     - 使用循环缓冲区机制               │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  3. 轨迹采样 (buffer.sample())        │
        │     - 随机采样一批episode              │
        │     - 可选：过滤、截断                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  4. 网络训练 (learner.train())        │
        │     - 计算当前Q值（主网络）            │
        │     - 计算目标Q值（目标网络）          │
        │     - 计算TD误差和损失                 │
        │     - 反向传播更新参数                 │
        │     - 定期更新目标网络                 │
        └───────────────────────────────────────┘
```

---

## 六、关键代码文件总结

| 功能 | 文件路径 | 关键方法 |
|------|---------|---------|
| **主训练循环** | `run.py` | `run()` 函数中的while循环 |
| **轨迹收集** | `runners/episode_runner.py` | `run()` 方法 |
| **动作选择** | `controllers/basic_controller.py` | `forward()`, `select_actions()` |
| **轨迹存储** | `components/episode_buffer.py` | `insert_episode_batch()` |
| **轨迹采样** | `components/episode_buffer.py` | `sample()` |
| **网络训练** | `learners/q_learner.py` | `train()` |
| **下层Q网络** | `modules/agents/agent.py` | Agent类的forward方法 |

---

## 七、训练参数配置

在配置文件中可以设置：

```yaml
# 训练相关
batch_size: 32              # 每次训练的batch大小
training_iters: 1            # 每次采样后训练的迭代次数
buffer_size: 32              # 经验回放缓冲区大小
max_traj_len: 64             # 最大轨迹长度（截断）

# Q学习相关
gamma: 0.99                  # 折扣因子
double_q: True               # 是否使用Double Q-learning
target_update_interval: 200  # 目标网络更新间隔（episode数）

# 优化器相关
lr: 0.0005                   # 学习率
grad_norm_clip: 10           # 梯度裁剪
```

---

## 八、总结

ALMA的下层Q网络训练遵循标准的深度Q学习流程：

1. **收集阶段**：使用当前策略（epsilon-greedy）与环境交互，收集轨迹
2. **存储阶段**：将轨迹存储到经验回放缓冲区
3. **采样阶段**：从缓冲区随机采样一批轨迹
4. **训练阶段**：
   - 使用主网络计算当前Q值
   - 使用目标网络计算目标Q值
   - 计算TD误差和损失
   - 反向传播更新主网络参数
   - 定期将主网络参数复制到目标网络

这种设计实现了**离线学习**（off-policy learning），允许从历史经验中学习，提高样本效率。

