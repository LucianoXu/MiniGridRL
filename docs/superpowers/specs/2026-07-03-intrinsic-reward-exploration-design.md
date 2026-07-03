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

## 本次采用的默认(待用户确认,可推翻)

用户在澄清阶段暂时离开,以下三个分叉按最合理默认推进,已在设计中标注:

1. **接入点 = 先挂到现有 REINFORCE**。改动最小、最快验证 seam。接口设计成未来 `OnPolicyAgent`
   (A2C/PPO)也能原样复用。
2. **奖励合并 = 单一合并回报** `r_total = r_ext + β·r_int`,一条 return 流。REINFORCE 无 critic,
   RND 原版的"内在/外在双 value head + 各自 discount"与当前无 critic 结构不兼容,留到 A2C 阶段。
3. **spec 范围 = 抽象与三种方法全部完整设计**;代码按 **RND → ICM → LPM** 递增复杂度分阶段实现,
   每阶段独立可验证。

## 范围边界

**做:**
- `IntrinsicReward` 抽象基类(`compute_intrinsic` + `update` 两个核心方法 + checkpoint/设备)。
- 三种方法的完整设计:RND / ICM / LPM 的网络结构、奖励公式、自身 loss、内部状态。
- 共享工具:`RunningMeanStd`(内在奖励归一化,RND 强依赖)。
- REINFORCE 收集循环的最小改动:存 `(obs, act, next_obs)` trace,在有效步(`step_mask`)上算奖励和训练。
- factory / config 集成:`intrinsic_reward_factory(cfg)`,`agent.intrinsic_reward` 配置块,缺省 = 纯外在。
- TensorBoard 内在奖励诊断指标。

**不做(YAGNI):**
- 双 value head / 内在-外在分离 discount —— 留到 A2C/PPO 阶段。
- episodic bonus 类方法(NGU / EDT / count-based)—— 本批三种都是 global 非 episodic。
- noisy-TV 环境注入 wrapper —— 见"未来"一节,LPM 的优势要靠它才显现,列为紧接着的后续项。
- RolloutBuffer 大重构 —— 见下文,本次只做 REINFORCE 局部改动,但接口对齐未来 buffer。

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

    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...
    def to(self, device) -> "IntrinsicReward": ...
```

**为什么是两个方法而非一个 `process(...)`**:强制正确时序。agent 主循环里顺序固定为
`compute_intrinsic → 用 r_total 做 RL 更新 → update`。此序对 RND/ICM 无所谓,对 **LPM 是必须**的
(reward 用当前 dynamics `f^(τ-1)` 算,`update` 之后才变成 `f^(τ)`)。

**编码器复用**:三种方法都需要把 7×7×3 uint8 编码成向量,复用已有的
`models/interface.py:GridEmbedding`。**注意 RND 的 target 必须是独立、固定、随机初始化**的网络,
不能与策略或预测器共享参数。

## 各方法设计

### RND(第 1 阶段)

- 两个网络 `target`(冻结随机)、`predictor`(可训),结构相同:`GridEmbedding → MLP → R^k`(k≈128)。
- obs 归一化:对输入做 running mean/std 标准化并 clip(RND 对此强依赖)。
- `compute_intrinsic`:`r_int = ‖predictor(s') − target(s')‖²`(仅用 `next_obs`),
  再除以内在奖励的 running std(`RunningMeanStd`)做归一化,让 β 可跨环境调。
- `update`:对同批 `s'` 最小化 `‖predictor − target‖²`(target 不回传梯度)。自带 Adam。

### ICM(第 2 阶段)

- 编码器 `φ`:`GridEmbedding → MLP → R^m`。
- inverse 模型:`[φ(s), φ(s')] → logits over 7 actions`,损失 = 交叉熵(MiniGrid 是 Discrete(7),天然契合)。
- forward 模型:`[φ(s), onehot(a)] → φ̂(s')`,损失 = `½‖φ̂(s') − φ(s').detach()‖²`。
- `compute_intrinsic`:`r_int = ½‖φ̂(s') − φ(s')‖²`(forward 误差)。
- `update`:`loss = (1−λ)·inverse_CE + λ·forward_MSE`,`λ` 可配(论文默认偏向 forward)。
  编码器主要经 inverse 损失学习"可控特征",从而对与动作无关的噪声不敏感。

### LPM(第 3 阶段,最复杂)

映射到 on-policy 设定:环境步 `t`,模型更新步 `τ` 对齐"每次 RL 更新一次"(每个 rollout 一次)。

- dynamics 模型 `f_θ`:`(o_t, a_t) → ô_{t+1}`(编码器 + 以 `onehot(a)` 为条件的解码/预测头)。
- error 模型 `g_ψ`:`(o_t, a_t) → R`,预测**上一轮** dynamics 的期望 log-MSE。
- 两个 buffer:replay buffer `B`(存 transition,拟合 `f`);定长队列 `D`(size d,存 `(o_t,a_t,ε)`,拟合 `g`)。
- 逐步误差:`ε_t = log MSE(o_{t+1}, f_θ(o_t,a_t))`,用当前(=本轮更新前 = `f^(τ-1)`)模型算。
- `compute_intrinsic`:`r_int = g_ψ(o_t,a_t) − ε_t`(若 `|D|=d` 否则 0,warmup);同时把
  `(o,a,o')→B`、`(o,a,ε)→D`。
- `update`(RL 更新之后):用 `B` 更新 `f_θ → f^(τ)`;用 `D` 更新 `g_ψ` 去拟合队列里的 ε
  (这些 ε 是 `f^(τ-1)` 的误差)。于是下轮 `r_int = g^(τ)(预测的旧模型误差) − 新模型误差`,
  模型有改进则为正 = learning progress。
- 关键正确性点(论文 Thm 4.2):`g` 必须拟合**期望**误差(用队列 D 而非单点保存旧模型),否则内在奖励
  可能在 IG>0 时变负,破坏与信息增益的单调关系。实现里体现为"`g` 拟合 D 中一批 ε 的回归",而非
  "存一份旧 `f` 直接推理"。

## 集成:REINFORCE 收集循环改动

当前 `REINFORCE.train`(`agents/REINFORCE.py`)的收集循环**只存了 reward / log_prob / entropy,没存
obs trace**。内在奖励需要 `(s_t, a_t, s_{t+1})` 三元组,所以:

1. 收集时新增存储:每步的 `obs`(即 `s_t`)、`acts`(`a_t`),下一步的 `obs` 即 `s_{t+1}`。
2. 尊重变长 episode 的 `step_mask`:只在有效步上抽出 transition 送去算奖励与训练(把有效步 flatten 成 `(N, …)`)。
3. 主循环顺序:`compute_intrinsic` 得到 `r_int` → `r_total = r_ext + β·r_int` → 用 `r_total` 走现有
   reward-to-go / baseline / surrogate loss → `intrinsic.update(...)` → log `intrinsic` 指标。
4. `intrinsic is None` 时完全走现有纯外在路径(零行为变化)。

**与未来 RolloutBuffer 的关系**:一个规规矩矩存 `(obs, act, next_obs, rew, done)` 的 buffer 能让内在奖励
几乎零成本挂上,A2C/PPO 也直接受益。本次**不做**大重构,只在 REINFORCE 内做局部 trace 存储;但
`compute_intrinsic/update` 的签名以"一批 transition"为单位,正好对齐未来 buffer 的取数方式,不会返工。

## config / factory 集成

沿用现有工厂模式(`agents/factory.py`、`models/factory.py`):

```yaml
agent:
  id: REINFORCE
  # ... 现有字段 ...
  intrinsic_reward:          # 省略 => 纯外在
    id: RND                  # RND | ICM | LPM
    beta: 0.5
    lr: 0.0001
    # 各方法特有字段(如 RND: feature_dim; ICM: lambda; LPM: queue_size, update_cycle)
```

- 新增 `minigridrl/intrinsic/factory.py:intrinsic_reward_factory(cfg) -> IntrinsicReward | None`。
- `agent_factory` 读到 `intrinsic_reward` 块则构造并注入 REINFORCE(新增可选构造参数 `intrinsic=None`)。
- 内在模块的 checkpoint 随 agent 一起 save/load(在 `save_local/load_local` 里加 `intrinsic.state_dict()`)。

## TensorBoard 指标

复用 `interface.py:tensorboard_write` 的 `None`=N/A 约定,新增:
`intrinsic_reward_mean`、`intrinsic_reward_std`、方法自身 loss(`rnd_loss` / `icm_forward_loss` /
`icm_inverse_loss` / `lpm_dynamics_loss` / `lpm_error_loss`)、`beta`。非当前方法的指标传 `None` 跳过。

## 模块布局

```
minigridrl/intrinsic/
  __init__.py
  interface.py     # IntrinsicReward 抽象基类 + RunningMeanStd 工具
  factory.py       # intrinsic_reward_factory(cfg)
  rnd.py           # RND            (阶段 1)
  icm.py           # ICM            (阶段 2)
  lpm.py           # LPM            (阶段 3)
```

## 实现顺序与验证

1. **阶段 1 — 抽象 + RND**:打通 REINFORCE 集成 seam。验证:Empty 环境上开/关 RND 都能训练,
   TensorBoard 出现 intrinsic 指标,`intrinsic=None` 与改动前行为一致。
2. **阶段 2 — ICM**:仅新增 `icm.py` + config 分支,不动 seam。
3. **阶段 3 — LPM**:实现双 buffer + 更新周期 + error model。验证单调性直觉:模型改进时 `r_int` 为正。
4. **对照实验**:在稀疏 MiniGrid(如 Empty-8x8 / DoorKey)上对比 无内在 / RND / ICM / LPM 的探索效率。

## 已知简化与未来

- **单一合并回报**:未区分内在/外在的 discount 与 value(RND 原版做法),等 A2C/critic 后再升级。
- **noisy-TV wrapper(紧接后续)**:无噪声时 RND/ICM/LPM 表现接近;要复现 LPM 的核心卖点,需一个向
  MiniGrid 观测注入 state-noise / action-triggered-noise 的 wrapper。列为本设计之后的第一个后续项。
- **episodic bonus 家族**:接口以 transition 为单位,未来要支持 episodic memory 需扩展(非本次)。

## 开放问题(待用户确认)

1. 上面三个默认(接入点 REINFORCE / 单一合并回报 / 三种全设计-分阶段实现)是否接受?
2. LPM 的更新周期 `N` 对齐"每 rollout 一次"是否 OK,还是要支持更细的按环境步周期?
3. 是否要把 noisy-TV wrapper 纳入本 spec,还是确实拆成紧接的下一个立项?
