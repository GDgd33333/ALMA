# `'task_allocation': 'heuristic'` 在 savethecity 环境中的代码执行路径

## 概述

当配置 `'task_allocation': 'heuristic'` 时，系统会使用基于距离的启发式算法来分配智能体到任务（建筑），而不是使用学习到的分配策略。

## 代码执行路径

### 1. 初始化阶段 (`run.py`)

**文件**: `/data/gu-di/ALMA/src/run.py`

**位置**: 第 192-193 行

```python
if args.hier_agent["task_allocation"] == 'heuristic':
    args.env_args['heuristic_alloc'] = True
```

**作用**: 
- 检测到 `task_allocation` 设置为 `'heuristic'`
- 将 `heuristic_alloc=True` 传递给环境参数

---

### 2. 环境初始化 (`firefighters.py`)

**文件**: `/data/gu-di/ALMA/src/envs/firefighters/firefighters.py`

**位置**: 第 256-268 行

```python
class FireFightersEnv(MultiAgentEnv):
    def __init__(self,
                 entity_scheme=True,
                 heuristic_alloc=False,  # 接收参数
                 ...):
        self.heuristic_alloc = heuristic_alloc  # 保存标志
```

**作用**: 
- 环境接收 `heuristic_alloc` 参数
- 保存为实例变量，用于后续判断

---

### 3. Controller 初始化 (`basic_controller.py`)

**文件**: `/data/gu-di/ALMA/src/controllers/basic_controller.py`

**位置**: 第 14 行

```python
self.heuristic_alloc = args.hier_agent['task_allocation'] == 'heuristic'
```

**作用**: 
- Controller 检测是否使用启发式分配
- 设置 `self.heuristic_alloc = True`

---

### 4. 每步执行：获取启发式分配 (`firefighters.py`)

**文件**: `/data/gu-di/ALMA/src/envs/firefighters/firefighters.py`

**位置**: 第 627-660 行（`get_masks` 方法中）

**关键代码**:

```python
def get_masks(self):
    # ... 其他代码 ...
    
    entity2task = np.ones((self.max_n_agents + self.max_n_buildings,
                          self.max_n_buildings), dtype=np.uint8)
    
    if self.heuristic_alloc:  # 如果启用启发式分配
        # 1. 获取所有活跃建筑（未完成且未烧毁）
        active_blds = [
            bld for bld in self.buildings
            if not (bld.complete or bld.burned_down)
        ]
        
        # 2. 对每个智能体计算分配
        for ai, agent in enumerate(self.agents):
            # 2.1 计算到所有活跃建筑的曼哈顿距离
            dist_blds = [
                (abs(agent.x - bld.x) + abs(agent.y - bld.y), bld)
                for bld in active_blds
            ]
            
            # 2.2 按距离排序（最近的在前）
            sorted_blds = [bld for _, bld in sorted(dist_blds, key=lambda x: x[0])]
            
            # 2.3 分类建筑
            fire_blds = [bld for bld in sorted_blds if bld.fire_strength > 0.0]  # 着火建筑
            not_fire_blds = [bld for bld in sorted_blds if bld.fire_strength == 0.0]  # 未着火建筑
            
            # 2.4 根据智能体类型确定优先级
            if agent.ent_id == F_ID:  # 消防员
                priority_blds = fire_blds + not_fire_blds  # 优先灭火
            elif agent.ent_id == B_ID:  # 建造者
                priority_blds = not_fire_blds + fire_blds  # 优先建造
            elif agent.ent_id == G_ID:  # 通用智能体
                priority_blds = fire_blds + not_fire_blds  # 优先灭火（可以冻结）
            
            # 2.5 分配智能体到最近的优先建筑
            if len(priority_blds) > 0:
                assgn_bld_id = priority_blds[0].obj_id - self.n_agents
                entity2task[ai, assgn_bld_id] = 0  # 0 表示分配，1 表示未分配
    
    # 3. 建筑总是分配到自己的任务
    for bi in range(self.n_buildings):
        entity2task[self.max_n_agents + bi, bi] = 0
    
    masks['entity2task_mask'] = entity2task
    return masks
```

**启发式算法逻辑**:

1. **距离计算**: 使用曼哈顿距离 `|x1-x2| + |y1-y2|`
2. **建筑分类**: 
   - `fire_blds`: 正在着火的建筑
   - `not_fire_blds`: 未着火的建筑
3. **优先级规则**:
   - **消防员 (F_ID)**: `fire_blds + not_fire_blds` → 优先灭火
   - **建造者 (B_ID)**: `not_fire_blds + fire_blds` → 优先建造（需要未着火才能建造）
   - **通用智能体 (G_ID)**: `fire_blds + not_fire_blds` → 优先灭火（可以冻结）
4. **分配规则**: 每个智能体分配到**最近的优先建筑**

---

### 5. Controller 使用启发式分配 (`basic_controller.py`)

**文件**: `/data/gu-di/ALMA/src/controllers/basic_controller.py`

**位置**: 第 39-41 行（`select_actions` 方法中）

**关键代码**:

```python
def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
    # ... 其他代码 ...
    
    if self.use_alloc or self.use_copa:
        decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
        if decision_pts.sum() >= 1:  # 在决策点
            if self.use_alloc:
                if self.learned_alloc:
                    # 学习到的分配（AQL）
                    meta_batch = self._make_meta_batch(ep_batch, t_ep)
                    new_allocs = self.compute_allocation(meta_batch, ...)
                elif self.heuristic_alloc:  # ← 这里！
                    # 启发式分配：从环境计算的 entity2task_mask 中提取
                    new_allocs = 1 - ep_batch['entity2task_mask'][:, t_ep, :self.n_agents][decision_pts == 1]
                elif self.random_alloc:
                    # 随机分配
                    new_allocs = random_allocs(...)
                
                # 更新任务分配
                self.task_allocations[decision_pts == 1] = new_allocs.to(self.task_allocations.dtype)
    
    # ... 后续使用分配进行动作选择 ...
```

**关键点**:
- `entity2task_mask` 中，`0` 表示分配，`1` 表示未分配
- 所以需要 `1 - entity2task_mask` 来转换为分配矩阵（`1` 表示分配，`0` 表示未分配）
- 启发式分配在**每个决策点**从环境的 `entity2task_mask` 中提取

---

## 完整执行流程

```
1. run.py (初始化)
   └─> 检测 task_allocation == 'heuristic'
       └─> 设置 args.env_args['heuristic_alloc'] = True

2. FireFightersEnv.__init__()
   └─> 接收 heuristic_alloc 参数
       └─> 保存为 self.heuristic_alloc

3. BasicMAC.__init__()
   └─> 设置 self.heuristic_alloc = True

4. 每个时间步 (step):
   │
   ├─> FireFightersEnv.get_masks()
   │   └─> 如果 self.heuristic_alloc == True:
   │       ├─> 获取活跃建筑列表
   │       ├─> 对每个智能体:
   │       │   ├─> 计算到所有活跃建筑的曼哈顿距离
   │       │   ├─> 按距离排序
   │       │   ├─> 分类建筑（着火/未着火）
   │       │   ├─> 根据智能体类型确定优先级
   │       │   └─> 分配到最近的优先建筑
   │       └─> 将分配结果存储在 entity2task_mask 中
   │
   └─> BasicMAC.select_actions()
       └─> 在决策点 (hier_decision == 1):
           └─> 如果 self.heuristic_alloc == True:
               └─> 从 ep_batch['entity2task_mask'] 中提取分配
                   └─> new_allocs = 1 - entity2task_mask[:, :n_agents]
                       └─> 更新 self.task_allocations
                           └─> 用于后续的动作选择
```

---

## 关键数据结构

### `entity2task_mask`
- **形状**: `(max_n_agents + max_n_buildings, n_buildings)`
- **含义**: 
  - `0`: 实体分配到该任务
  - `1`: 实体未分配到该任务
- **在启发式模式下**: 由环境在 `get_masks()` 中计算

### `task_allocations`
- **形状**: `(batch_size, n_agents, n_tasks)`
- **含义**: 
  - `1`: 智能体分配到该任务
  - `0`: 智能体未分配到该任务
- **在启发式模式下**: 从 `entity2task_mask` 转换而来

---

## 与学习分配 (AQL) 的区别

| 特性 | Heuristic | AQL (学习分配) |
|------|----------|----------------|
| **计算位置** | 环境 (`get_masks`) | Controller (`compute_allocation`) |
| **更新频率** | 每步都更新 | 仅在决策点更新 |
| **算法** | 基于距离的启发式 | 神经网络学习 |
| **参数** | 无（硬编码规则） | 可训练参数 |
| **适应性** | 固定规则 | 可适应环境 |

---

## 总结

当 `'task_allocation': 'heuristic'` 时：

1. **环境负责计算分配**: 在 `get_masks()` 中，基于曼哈顿距离和智能体类型计算分配
2. **Controller 提取分配**: 在决策点从 `entity2task_mask` 中提取分配结果
3. **无需训练**: 启发式分配是硬编码的规则，不需要训练分配网络
4. **每步更新**: 分配在每个时间步都会重新计算，确保智能体总是分配到最近的优先建筑

这种设计使得系统可以在不使用学习分配的情况下运行，适合作为基线或调试工具。

