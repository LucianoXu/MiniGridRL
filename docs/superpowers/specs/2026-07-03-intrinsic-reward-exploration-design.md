# 内在奖励驱动探索(Intrinsic Reward)— 设计文档

日期:2026-07-03
状态:待用户确认

## 背景与目标

在 MiniGrid 这类稀疏奖励环境里,纯随机探索效率极低。本设计引入**内在奖励驱动的探索**:
在环境外在奖励 `r_ext` 之外,由 agent 自身产生一个内在奖励 `r_int` 作为探索信号,合成
`r_total = r_ext + β·r_int` 供 RL 更新使用。

目标是搭一个统一的 `IntrinsicReward` 抽象,让不同内在奖励方法可插拔替换,首批覆盖三种:

- **RND**(Random Network Distillation,Burda et al. 2018)—— 预测误差 / 新颖度。
- **ICM**(Intrinsic Curiosity Module,Pathak et al. 2017)—— 可控特征空间的 forward 预测误差。
- **LPM**(Learning Progress Monitoring,Hou et al. ICLR 2026,`references/Hou2026_LPM_BeyondNoisyTVs.pdf`)
  —— 奖励"模型改进量"而非预测误差,对 noisy-TV(不可学习的随机性)天然鲁棒。

LPM 论文附录 B 正是在 **MiniGrid(Lava Crossing)+ PPO** 上对比 ICM vs LPM,所以本 playground
就是它的合法测试床。核心关注点承自 `SPEC.md`:可观测性、可扩展性、可复现性。

## 本次采用的决定(用户已确认)

1. **接入点 = 现有 PPO**(用户指定,不在 REINFORCE 上做)。PPO(`agents/PPO.py`)已具备完整的
   time-major rollout buffer 与 critic,是内在奖励最自然的宿主;LPM 论文附录 B 也正是 PPO+GAE 设置。
   接口以"一批 transition"为单位,未来 A2C 或其他 on-policy agent 也能原样复用。
2. **奖励合并 = 单一合并回报** `r_total = r_ext + β·r_int`,一条 return 流:把 `β·r_int` 加进
   `buffers["rew"]` 后照常走 GAE,单 critic 学合成回报的 value。这与 LPM 论文的 MiniGrid+PPO 设置一致。
   PPO 已有 critic,故 RND 原版的"内在/外在双 value head + 各自 discount"**未来可选升级**,本次不做。
3. **spec 范围 = 抽象与三种方法全部完整设计**;代码按 **RND → ICM → LPM** 递增复杂度分阶段实现,
   每阶段独立可验证。

## 范围边界

**做:**
- `IntrinsicReward` 抽象基类(`compute_intrinsic` + `update` 两个核心方法 + checkpoint/设备)。
- 三种方法的完整设计:RND / ICM / LPM 的网络结构、奖励公式、自身 loss、内部状态。
- 共享工具:`RunningMeanStd`(内在奖励归一化,RND 强依赖)。
- PPO 训练循环的最小改动:从已有 rollout buffer 取出有效 transition,合成 `r_total` 后再走 GAE;
  RL 更新后调 `intrinsic.update`。缺省(无 intrinsic)行为与改动前完全一致。
- factory / config 集成:`intrinsic_reward_factory(cfg)`,`agent.intrinsic_reward` 配置块,缺省 = 纯外在。
- TensorBoard 内在奖励诊断指标;内在模块随 PPO checkpoint 一起 save/load。

**不做(YAGNI):**
- 双 value head / 内在-外在分离 discount —— PPO 已有 critic,留作未来可选升级。
- episodic bonus 类方法(NGU / EDT / count-based)—— 本批三种都是 global 非 episodic。
- noisy-TV 环境注入 wrapper —— 见"未来"一节,LPM 的优势要靠它才显现,列为紧接着的后续项。

## 三种方法的统一坐标系

| | 输入 | 自带网络 | 内在奖励 `r_int` | 自身 loss | 内部状态 |
|---|---|---|---|---|---|
| RND | `s'` | 固定随机 target `f` + 预测器 `f̂` | `‖f̂(s')−f(s')‖²` | 同式(训 `f̂`→`f`) | obs/reward running std |
| ICM | `s,a,s'` | 编码器 `φ` + forward + inverse | `½‖φ̂(s')−φ(s')‖²`(forward 误差) | `(1−λ)·inverse_CE + λ·forward_MSE` | 无 |
| LPM | `s,a,s'` | dynamics `f_θ` + error model `g_ψ` | `g_ψ(s,a) − ε_t`,`ε=log MSE` | `f` 拟合 buffer B,`g` 拟合误差队列 D | buffer B + 定长队列 D + 更新周期 τ |

抽象出的共性只有两件事:(1) 给一批 transition `(s,a,s')` 吐出 per-step 内在奖励(**detached**,不带
policy 梯度);(2) 用收集到的 transition 训练自身网络并返回诊断指标。所有差异都藏进实现。

## `IntrinsicReward` 接口

放在 `minigridrl/intrinsic/interface.py`。

```python
class IntrinsicReward(ABC):
    beta: float                       # r_total = r_ext + beta * r_int

    @abstractmethod
    def compute_intrinsic(
        self, obs: Tensor, action: Tensor, next_obs: Tensor
    ) -> Tensor:
        """批量 transition -> per-step 内在奖励 (detached, shape (N,))。
           收集阶段调用。可含副作用(如 RND 更新 reward running std、
           LPM 把误差压入队列 D)。绝不回传 policy 梯度。"""

    @abstractmethod
    def update(
        self, obs: Tensor, action: Tensor, next_obs: Tensor
    ) -> dict[str, float]:
        """用收集到的有效 transition 训练自身网络,返回 loss 指标供 TensorBoard。
           契约:必须在同一批数据的 compute_intrinsic 之后调用
           (LPM 依赖'先用当前模型算奖励、再更新模型'的时序)。"""
```

接口只有 `beta` 属性 + 两个方法。**checkpoint 靠整体 pickle**:PPO 的 `save_local` 已经用
`weights_only=False` pickle 了整个 `policy`/`value_net` 对象,内在模块(持有 nn.Module + Adam +
running-stats,均可 pickle)照此整体存取即可,无需单独 `state_dict`。

**为什么是两个方法而非一个 `process(...)`**:强制正确时序。agent 主循环里顺序固定为
`compute_intrinsic → 用 r_total 做 RL 更新 → update`。此序对 RND/ICM 无所谓,对 **LPM 是必须**的
(reward 用当前 dynamics `f^(τ-1)` 算,`update` 之后才变成 `f^(τ)`)。

**编码器复用**:三种方法都需要把 7×7×3 uint8 编码成向量,复用已有的
`models/interface.py:GridEmbedding`。**注意 RND 的 target 必须是独立、固定、随机初始化**的网络,
不能与策略或预测器共享参数。

## 各方法设计

### RND(第 1 阶段)

- 两个网络 `target`(冻结随机)、`predictor`(可训),结构相同:`GridEmbedding → MLP → R^k`(k≈128),
  用共享的 `GridEncoder`(见下"共享工具")。
- **obs 归一化的调整**:RND 原版对像素输入做 running mean/std 标准化;但本项目 obs 经 `GridEmbedding`
  的整数索引查表(需整数),无法在查表前标准化整数索引。故**放弃对原始 obs 的标准化**(MiniGrid 取值小且
  有界,embedding 已承担表示),仅保留下面的**内在奖励归一化**。
- `compute_intrinsic`:`r_int = ‖predictor(s') − target(s')‖²`(逐样本对特征维求均值,仅用 `next_obs`),
  再除以内在奖励的 running std(`RunningMeanStd`,shape=())做归一化,让 β 可跨环境调。
- `update`:对同批 `s'` 最小化 `‖predictor − target.detach()‖²`。自带 Adam。

### ICM(第 2 阶段)

- 编码器 `φ`:`GridEmbedding → MLP → R^m`。
- inverse 模型:`[φ(s), φ(s')] → logits over 7 actions`,损失 = 交叉熵(MiniGrid 是 Discrete(7),天然契合)。
- forward 模型:`[φ(s), onehot(a)] → φ̂(s')`,损失 = `½‖φ̂(s') − φ(s').detach()‖²`。
- `compute_intrinsic`:`r_int = ½‖φ̂(s') − φ(s')‖²`(forward 误差)。
- `update`:`loss = (1−λ)·inverse_CE + λ·forward_MSE`,`λ` 可配(论文默认偏向 forward)。
  编码器主要经 inverse 损失学习"可控特征",从而对与动作无关的噪声不敏感。

### LPM(第 3 阶段,最复杂)

映射到 on-policy 设定:环境步 `t`,模型更新步 `τ` 对齐"每次 RL 更新一次"(每个 rollout 一次)。

- **MiniGrid 适配**:论文 dynamics 预测原始下一观测;这里 obs 是 7×7×3 整数栅格(类别型,直接回归/解码不自然)。
  改为在**固定随机编码器** `ψ`(冻结 `GridEncoder`,类比 RND target)的特征空间里做预测——`ψ` 固定保证
  target 不塌缩、`ε` 有定义。
- dynamics 模型 `f_θ`:`[ψ(o_t), onehot(a_t)] → ψ̂(o_{t+1})`(在 `ψ` 特征空间预测下一特征)。
- error 模型 `g_ψ`:`[ψ(o_t), onehot(a_t)] → R`,预测**上一轮** dynamics 的期望 log-MSE。
- 两个 buffer:replay buffer `B`(存 transition,拟合 `f`);定长队列 `D`(size d,存 `(o_t,a_t,ε)`,拟合 `g`)。
- 逐步误差:`ε_t = log MSE(ψ(o_{t+1}), f_θ([ψ(o_t),onehot(a_t)]))`,用当前(=本轮更新前 = `f^(τ-1)`)模型算。
- `compute_intrinsic`:`r_int = g_ψ(o_t,a_t) − ε_t`(若 `|D|=d` 否则 0,warmup);同时把
  `(o,a,o')→B`、`(o,a,ε)→D`。
- `update`(RL 更新之后):用 `B` 更新 `f_θ → f^(τ)`;用 `D` 更新 `g_ψ` 去拟合队列里的 ε
  (这些 ε 是 `f^(τ-1)` 的误差)。于是下轮 `r_int = g^(τ)(预测的旧模型误差) − 新模型误差`,
  模型有改进则为正 = learning progress。
- 关键正确性点(论文 Thm 4.2):`g` 必须拟合**期望**误差(用队列 D 而非单点保存旧模型),否则内在奖励
  可能在 IG>0 时变负,破坏与信息增益的单调关系。实现里体现为"`g` 拟合 D 中一批 ε 的回归",而非
  "存一份旧 `f` 直接推理"。

## 集成:PPO 训练循环改动

好消息:PPO(`agents/PPO.py`)的 `_collect_rollout` **已经**把 `obs / act / rew / valid` 等按
time-major `(T, N, …)` 存进了 buffer,并返回 rollout 结束后的 carry-over `obs`。内在奖励需要的
`(s_t, a_t, s_{t+1})` 三元组几乎是现成的,**无需 RolloutBuffer 重构**。

**取 `next_obs`(关键正确性点)**:对时间步 `t`,`next_obs[t] = obs[t+1]`(`t < T-1`),
`next_obs[T-1] = carry_over_obs`。在 NEXT_STEP autoreset 下这是正确的:当 `done[t]=True`,`obs[t+1]`
恰是该 episode 的**真实终止观测**(PPO 的 GAE 注释已依赖这一语义来 bootstrap 截断),而紧随其后的
autoreset dummy 步由 `valid=False` 屏蔽。因此我们**只在 `valid` 步上**计算内在奖励与训练,`next_obs`
取 `obs[t+1]` 始终对应正确的转移。

`PPO.train` 的改动(仅在 `intrinsic is not None` 时生效,否则零行为变化):

1. `buffers, obs, prev_done, stats = self._collect_rollout(...)` 之后、`self._update(buffers)` 之前,
   插入一个**批量内在奖励 pass**:
   - 构造 `next_obs` buffer(如上);把 `valid` 的 `(obs, act, next_obs)` flatten 成 `(M, …)`。
   - `r_int = intrinsic.compute_intrinsic(obs_v, act_v, next_obs_v)`(detached)。
   - 把 `β·r_int` 按 valid 位置**加回** `buffers["rew"]`,得到合成回报。
2. `self._update(buffers)` 照常:GAE 用合成回报计算,critic 学合成回报的 value,PPO clip 更新。
3. `_update` 之后:`intrinsic.update(obs_v, act_v, next_obs_v)`,返回指标写 TensorBoard。
4. **时序**:`compute_intrinsic`(收集后、更新前,用当前内在模型)→ PPO `_update`(不动内在模型)→
   `intrinsic.update`。这满足 LPM"先用当前 dynamics 算奖励、再更新模型"的硬约束。PPO 每个 rollout
   恰好一次收集 + 一次更新,天然对齐 LPM 的模型更新步 `τ`。

内在奖励做成**批量 pass**(而非塞进 `_collect_rollout` 逐步计算):内在模型在一个 rollout 内不变,批量
等价且更高效,也让 `_collect_rollout` 保持内在无关、干净。

**checkpoint**:`save_local/load_local` 增加 `intrinsic.state_dict()` 的存取(仅当存在)。

## config / factory 集成

沿用现有工厂模式(`agents/factory.py`、`models/factory.py`):

```yaml
agent:
  id: PPO
  # ... 现有 PPO 字段(lr / gamma / clip_range / n_epochs / value_model ...) ...
  intrinsic_reward:          # 省略 => 纯外在
    id: RND                  # RND | ICM | LPM
    beta: 0.5
    lr: 0.0001
    # 各方法特有字段(如 RND: feature_dim; ICM: lambda; LPM: queue_size, update_cycle)
```

- 新增 `minigridrl/intrinsic/factory.py:intrinsic_reward_factory(cfg) -> IntrinsicReward | None`。
- `agent_factory` 的 PPO 分支读到 `intrinsic_reward` 块则构造并注入 PPO(`PPO.__init__` 新增可选参数
  `intrinsic: IntrinsicReward | None = None`);缺省 = 纯外在。
- 内在模块的 checkpoint 随 agent 一起 save/load(在 PPO `save_local/load_local` 里加 `intrinsic.state_dict()`)。

## TensorBoard 指标

复用 `interface.py:tensorboard_write` 的 `None`=N/A 约定,在 PPO 现有指标之外新增:
`intrinsic_reward_mean`、`intrinsic_reward_std`、方法自身 loss(`rnd_loss` / `icm_forward_loss` /
`icm_inverse_loss` / `lpm_dynamics_loss` / `lpm_error_loss`)、`beta`。无 intrinsic 或非当前方法的指标传
`None` 跳过。

## 模块布局

```
minigridrl/intrinsic/
  __init__.py
  interface.py     # IntrinsicReward 抽象基类 + RunningMeanStd + build_mlp + GridEncoder(复用 GridEmbedding)
  factory.py       # intrinsic_reward_factory(cfg)
  rnd.py           # RND            (阶段 1)
  icm.py           # ICM            (阶段 2)
  lpm.py           # LPM            (阶段 3)
```

## 实现顺序与验证

1. **阶段 1 — 抽象 + RND**:打通 PPO 集成 seam。验证:`intrinsic=None` 时训练与改动前逐指标一致
   (回归保护);开 RND 后 Empty 环境能训练,TensorBoard 出现 intrinsic 指标。
2. **阶段 2 — ICM**:仅新增 `icm.py` + config 分支,不动 seam。
3. **阶段 3 — LPM**:实现双 buffer + 更新周期 + error model。验证单调性直觉:模型改进时 `r_int` 为正。
4. **对照实验**:在稀疏 MiniGrid(如 Empty-8x8 / DoorKey)上对比 无内在 / RND / ICM / LPM 的探索效率。

## 已知简化与未来

- **单一合并回报**:未区分内在/外在的 discount 与 value(RND 原版做法)。PPO 已有 critic,升级到双
  value head(内在非 episodic、独立 discount)是纯增量,列为未来可选项。
- **noisy-TV wrapper(紧接后续)**:无噪声时 RND/ICM/LPM 表现接近;要复现 LPM 的核心卖点,需一个向
  MiniGrid 观测注入 state-noise / action-triggered-noise 的 wrapper。列为本设计之后的第一个后续项。
- **episodic bonus 家族**:接口以 transition 为单位,未来要支持 episodic memory 需扩展(非本次)。

## 开放问题(待用户确认)

三个主要分叉已确认(接入点 = PPO / 单一合并回报 / 三种全设计-分阶段实现)。剩余待确认:

1. LPM 的更新周期 `N` 对齐"每 rollout 一次"是否 OK,还是要支持更细的按环境步周期?
2. noisy-TV wrapper 确认拆成紧接的下一个立项(不纳入本 spec)?
3. RND 的 obs 归一化用 running mean/std;是否要与后续可能的 `VecNormalize` 类 env-level 归一化统一,
   还是内在模块自持一份即可(本设计取后者)?
