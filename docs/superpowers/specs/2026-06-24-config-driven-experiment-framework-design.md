# 配置驱动实验框架 — 设计文档

日期:2026-06-24
状态:已确认,待实现规划

## 背景与目标

MiniGridRL 是一个在 [MiniGrid](https://minigrid.farama.org) 环境上探索 RL 方法的 playground。本设计参考
[QuantBeyondEfficiency](https://github.com/LucianoXu/QuantBeyoundEfficiency) 的模式:**用 YAML 配置文件驱动实验,
并以自描述的目录结构保存产出**。

本次范围**只搭配置驱动骨架**:`config_type` 体系、工厂分发、引用归一化、实验目录与文件保存约定。
训练后端标准化到 stable-baselines3(SB3),评估/可视化前端先用最小占位接入,后续各自单独立项。

设计的核心关注点(承自 `SPEC.md`):可观测性(observability)、可扩展性(extendability)、可复现性(reproducibility)。

## 范围边界

**做:**
- 四种 `config_type`(env / agent / experiment / matrix)与 `run()` 分发。
- 配置引用归一化(引用为主 + 允许内联)。
- 自描述实验目录与文件保存约定(`trials/` 原始、`results/` 精选)。
- env / agent 工厂(SB3 后端),设备自动探测。
- 最小可用的 train → eval → save 编排,验证骨架跑通。

**不做(YAGNI,留给后续 brainstorm):**
- 完整评估体系、可视化前端、trace 回放 —— 各自单独立项。
- from-scratch PPO 不再维护(SB3 为准);`main.py` 留作参考,不接入框架。
- CLI 入口 —— notebook 为主,CLI 后续可加。

## 入口

**Notebook 为主**。`cookbook.ipynb` 中:

```python
from minigridrl import run
run("configs/experiments/ppo_empty5x5.yaml")
```

`run()` 是核心库函数,接受 config 路径(str/Path)或已加载的 dict。CLI 入口非本次重点。

## 模块布局

```
src/minigridrl/
  __init__.py        # 导出 run
  run.py             # run(config): 按 config_type 分发
  config.py          # 配置解析/归一化:load yaml、把引用展开成内联、校验 config_type
  factory.py         # 顶层工厂:dispatch 到 env/agent 工厂
  envs/factory.py    # env_factory(cfg) -> gym.Env (env id + wrappers + obs 处理)
  agents/factory.py  # agent_factory(cfg, env) -> SB3 model (PPO/A2C/DQN…)
  experiment.py      # 单次实验编排:建目录、存配置/环境快照、tee 日志、train→eval→save
  utils.py           # (已有) yaml/json io、tee_console、collect_environment、seed_everything

configs/
  envs/empty5x5.yaml
  agents/ppo.yaml
  experiments/ppo_empty5x5.yaml
  matrix/ppo_sweep.yaml

cookbook.ipynb       # 主入口示例
```

## config_type 体系

每个 YAML 顶层带 `config_type` 字段。`run()` 据此分发:

| config_type  | 作用                                                                 | 类比 QuantBeyond     |
|--------------|----------------------------------------------------------------------|----------------------|
| `env`        | 单个 MiniGrid 环境规格(env id、wrapper、obs 处理),构建并返回 gym.Env(主要用于交互/调试) | `model`             |
| `agent`      | 单个算法+网络+超参(SB3 PPO/A2C/DQN…);在给定 env 下构建,通常作为 experiment 的嵌套部分使用 | (新增原子)           |
| `experiment` | env + agent + 训练/评估参数 → 一次可跑的实验                            | `model_bench`        |
| `matrix`     | 在一个文件里枚举多组 env × agent,展开成多个 experiment 循环跑          | `model_bench_matrix` |

分发逻辑:
- `env` → `env_factory(cfg)` 构建并返回环境。
- `agent` → 校验 +(在给定 env 下)构建 SB3 model。
- `experiment` → 完整 `run_experiment(resolved_cfg)`(见下)。
- `matrix` → 展开成多个 experiment,逐个跑;父目录存 `index.yaml`。

## 引用归一化(引用为主 + 允许内联)

`config.py` 提供 `resolve(cfg, base_dir)`:

experiment 配置里的 `env` / `agent` 字段——
- 是**字符串** → 当作(相对 `base_dir` 的)路径,load 那个 YAML 并内联进来;
- 是 **dict** → 直接采用。

归一化后得到一份完全展开内联的 `resolved` 配置,存进实验目录,保证目录可独立复现。
matrix 中每个枚举项同样走 resolve 得到各自的 resolved experiment 配置。

## 实验编排(`experiment.py`)

`run_experiment(resolved_cfg)` 顺序:

1. **建目录** `trials/<name>-<timestamp>/`(name 取配置里的 `name`,timestamp 保证不覆盖、可排序)。
2. **`seed_everything(seed)`**(seed 来自配置)。
3. **存档**:`config.resolved.yaml`(全展开内联)+ `environment.json`(`collect_environment()` 快照)。
4. **`tee_console(console.log)`** 包住后续全部 stdout/stderr。
5. **构建** `env = env_factory(cfg["env"])`,`model = agent_factory(cfg["agent"], env)`。
   - agent 工厂把 SB3 的 `tensorboard_log` 指到 `<expdir>/tensorboard/`。
6. **train** = `model.learn(total_timesteps, ...)`(参数来自配置)→ tensorboard 自动记曲线。
7. **eval**(占位):跑 N 个 episode 收集 reward,写 `metrics.json`(均值/方差)。完整 eval 后续单独立项。
8. **save** `checkpoints/final.zip`(SB3 原生 `model.save`)。

## 实验目录(自描述)

`trials/` 整体 git-ignore;`results/` 手动精选、git-tracked。

```
trials/<name>-<timestamp>/
  config.resolved.yaml     # 全展开内联,可独立复现
  environment.json         # 机器/库/加速器快照
  console.log              # 全部控制台镜像
  tensorboard/             # SB3 训练曲线
  checkpoints/final.zip    # SB3 模型
  metrics.json             # eval 汇总
results/                   # 手动从 trials 精选、git-tracked
```

matrix 展开时:
```
trials/<matrix_name>-<timestamp>/
  index.yaml               # 列出各子实验 + 其相对路径
  <sub_name>-<timestamp>/  # 每个子实验各自一个完整实验目录(结构同上)
  ...
```

## 设备与并行

- `agent_factory` 自动探测 `device`:cuda → mps → cpu(SB3 `device=` 参数),对应 SPEC「自动检测后端」。
- 配置可显式覆盖 `device`。
- 并行(SB3 `make_vec_env` / `n_envs`)放在 `env_factory`,本次骨架只接最小可用,深入调优后续再做。

## 配置文件示例(说明形态,非最终)

`configs/agents/ppo.yaml`:
```yaml
config_type: agent
algo: PPO
policy: MlpPolicy
hyperparams:
  learning_rate: 2.5e-4
  n_steps: 1024
  gamma: 0.99
```

`configs/experiments/ppo_empty5x5.yaml`:
```yaml
config_type: experiment
name: ppo_empty5x5
seed: 0
env: ../envs/empty5x5.yaml      # 引用;也可直接内联成 dict
agent: ../agents/ppo.yaml       # 引用;也可直接内联成 dict
train:
  total_timesteps: 100000
eval:
  n_episodes: 20
```

## 测试策略

- `config.py` 的 resolve:引用展开、内联、缺失文件报错、config_type 校验 —— 单元测试。
- 工厂:给定最小配置能构建出 env / SB3 model;device 探测逻辑 —— 单元测试。
- `run_experiment`:用极小 `total_timesteps` 跑一次,断言目录与各产出文件存在、`metrics.json` 结构正确 —— 集成测试(冒烟)。
- matrix 展开:断言子实验数量、`index.yaml` 内容。
