# RoboTaskLab

[English](#robotasklab) | [简体中文](#简体中文)

RoboTaskLab helps researchers getting started with agentic robotics develop
agents and skills in simulation. It includes benchmarks for task execution and
physical in-context learning (Physical ICL) across tasks.

You implement two files: `participant/agent.py` chooses skills and
parameters; `participant/skills.py` executes robot actions. The harness supplies
scenes, language goals, budgets, source context, scoring, trajectories and videos.

![Panda insertion reference rollout](assets/insertion.gif)

## Evaluation tracks

| Track | Measures |
|---|---|
| Agent | Task completion under randomized layouts and physical parameters |
| Physical ICL | Paired transfer from source experience to different task families with frozen agent and skill code |

The Physical ICL benchmark evaluates transfer from source-task trajectories and
failure experience to different target task families. Six context conditions
share matched target initializations while agent and skill code remain frozen.

Manufacturing tasks include round/square insertion, round/square nut placement
and stacking with a Panda robot. Parameters vary actuator gain, object mass and
scene-wide sliding friction. Observations include privileged state, RGB and
wrist readings. The included policy uses state feedback and analytic servo
calibration without a trained model. The harness also supports LIBERO pick/place
and bounded skill refinement.

## Installation

Use Python 3.12. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[manufacturing]"
python scripts/prepare_windows_preview.py
```

On Linux, activate with `source .venv/bin/activate` and skip the Windows setup
script. Reference simulations were run on Windows; Linux rendering is unverified.
The manufacturing extra does not require PyTorch, model weights or API keys.

## Run an agent

```sh
eah-bench --panel benchmarks/manufacturing-agent-v2.json --split evaluation --out runs/agent
```

Open `runs/agent/report.html`. Edit the two participant files during development,
then freeze them for evaluation. Each run needs a new output directory. A task
failure returns exit code 1 and retains its evidence.

## Run cross-task transfer

```sh
eah-bench --panel benchmarks/manufacturing-picl-v2.json --split development --out runs/source
eah-transfer --panel benchmarks/manufacturing-picl-v2.json --source-run runs/source --out runs/transfer
```

Open `runs/transfer/report.html`. The panel uses a source nut task and target
insertion/stacking tasks in two physical worlds. Six conditions compare matching
experience, action effects, language, unrelated physics, shuffled feedback and
no context. Each target starts with a new agent and matched initialization.

## Generate a panel

```sh
python scripts/generate_manufacturing.py --track physical_icl --source-family random --physical-axis combined --worlds 6 --layouts 3 --seed 42 --out benchmarks/study.json
python scripts/generate_manufacturing.py --track agent --instruction "把圆销插入圆孔并松开" --worlds 2 --seed 24 --out benchmarks/language.json
```

Each experiment fixes the source family and variant. Reusing source experience
requires matching implementation and dependency fingerprints, plus the metric
version. The generator supports explicit English/Chinese goals; freeze the
resulting JSON before comparison.
See [benchmark protocol](docs/benchmark.md) for observation/action contracts,
recovery behavior, source banks and LIBERO setup.

## Reference results

The recordings use version 0.6.0 with corrected contact friction.

| Experiment | Task success |
|---|---|
| Agent: five tasks, two worlds | 8/10 |
| Transfer: no context | 3/6 |
| Transfer: matching source or action-effect context | 4/6 |
| Transfer: language-only, unrelated physics or shuffled feedback | 3/6 each |
| Independent manufacturing-only install, Chinese goals | 1/2 |

Both source tasks failed to complete, but their recorded motion probes provide
valid interaction experience. These small reference runs use a rule controller.
The observed transfer comes from reusing servo calibration and does not establish
learned contact dynamics. The panels were used during development and are not
untouched research test sets.

Reports separate task completion, cost and transfer gain. Raw sampled wrist load
is diagnostic by default, with no universal force threshold or combined score.
See [metric definitions](docs/metrics.md) for scoring and interpretation.

Browse the included recordings locally:

```sh
python -m http.server 8771 --bind 127.0.0.1 --directory assets
```

Open `http://127.0.0.1:8771/`. Reports include successes, failures, source episodes
and integrity hashes.

## Extend the harness

Use `--agent examples/process_agent.py` for an external inference process.
`EAH_AGENT_COMMAND` selects a UTF-8 JSON worker with explicit state. Low-level
policies can be wrapped in `skills.execute` after converting their action frame
and units. Model revisions and inference budgets are the experiment owner's
responsibility. See [integrations](docs/integrations.md).

## Development

```sh
python -m pip install -e ".[dev]"
python -m pip install numpy==1.26.4 imageio imageio-ffmpeg pillow
python -m pytest -q
ruff check src tests scripts examples participant
ruff format --check src tests scripts examples participant
python scripts/check_repository.py
```

[Architecture](docs/architecture.md) · [Contributing and publishing](CONTRIBUTING.md) ·
[Changelog](CHANGELOG.md)

## License

Local code is MIT licensed. Robot/environment assets and dependencies retain
[their original attribution and licenses](THIRD_PARTY.md).

## 简体中文

RoboTaskLab 面向刚开始研究 agentic robotics 的研究者。你可以在仿真中开发 agent 和技能，评测任务完成情况，以及物理交互经验能否跨任务迁移。项目包含任务执行和物理上下文学习（Physical ICL）两条评测方向。

参与者主要修改两个文件：`participant/agent.py` 决定调用什么技能、使用什么参数；`participant/skills.py` 执行机器人动作。项目提供场景、语言目标、执行预算、源任务上下文、评分、轨迹和录像。

### 评测方向

| 方向 | 评测内容 |
|---|---|
| Agent | 在随机布局和物理参数下完成任务的能力 |
| Physical ICL | 冻结 agent 与技能代码，比较源任务经验对不同目标任务类别的影响 |

Physical ICL benchmark 使用源任务轨迹和失败经验构造上下文，在匹配的目标初始状态下比较六种条件：无上下文、匹配源经验、动作效果、仅语言、不同物理条件的经验，以及打乱反馈。

制造任务使用 Panda 机械臂，包含圆销/方销插孔、圆形/方形螺母放置和堆叠。可变化的参数包括驱动增益、物体质量和场景滑动摩擦系数。观测包含仿真特权状态、RGB 图像和腕部传感器读数。默认策略使用状态反馈与解析伺服标定，不依赖训练好的模型。项目也支持 LIBERO 抓取放置和有界技能参数改进。

### 安装

使用 Python 3.12。在 Windows 的项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[manufacturing]"
python scripts/prepare_windows_preview.py
```

Linux 用户使用 `source .venv/bin/activate` 激活环境，并跳过 Windows 兼容脚本。参考仿真在 Windows 上完成，Linux 渲染尚未验证。制造任务依赖不包含 PyTorch，也不需要模型权重或 API 密钥。

### 运行 agent

```sh
eah-bench --panel benchmarks/manufacturing-agent-v2.json --split evaluation --out runs/agent
```

运行后打开 `runs/agent/report.html` 查看录像和结果。开发时修改参与者文件，正式评测前冻结代码。每次运行使用新的输出目录；任务失败会返回退出码 1，并保留执行证据。

### 运行跨任务迁移实验

```sh
eah-bench --panel benchmarks/manufacturing-picl-v2.json --split development --out runs/source
eah-transfer --panel benchmarks/manufacturing-picl-v2.json --source-run runs/source --out runs/transfer
```

打开 `runs/transfer/report.html` 比较各条件。示例在两个物理世界中采集螺母任务经验，再迁移到插孔和堆叠任务。每个目标 episode 创建新的 agent，并使用配对的初始化状态。

### 随机任务与语言指令

```sh
python scripts/generate_manufacturing.py --track physical_icl --source-family random --physical-axis combined --worlds 6 --layouts 3 --seed 42 --out benchmarks/study.json
python scripts/generate_manufacturing.py --track agent --instruction "把圆销插入圆孔并松开" --worlds 2 --seed 24 --out benchmarks/language.json
```

每次实验固定源任务类别和变体。复用源经验时，代码实现、依赖指纹和指标版本必须匹配。生成器支持明确的中英文任务指令；开展对照实验前，应冻结生成的 JSON 配置。

观测与动作接口、恢复行为、源经验库及 LIBERO 安装方法见[评测协议](docs/benchmark.md)。

### 参考结果与适用范围

录像使用修正接触摩擦后的 0.6.0 版本生成。

| 实验 | 任务成功数 |
|---|---|
| Agent：五种任务、两个物理世界 | 8/10 |
| 跨任务迁移：无上下文 | 3/6 |
| 跨任务迁移：匹配源经验或动作效果上下文 | 4/6 |
| 跨任务迁移：仅语言、不同物理条件经验或打乱反馈 | 各 3/6 |
| 独立安装制造任务依赖，使用中文目标 | 1/2 |

两条源任务均未完成，但保留了有效的动作探测数据。以上是规则控制器的小规模参考实验，迁移增益来自伺服标定经验的复用，不能据此证明模型学会了接触动力学。这些任务配置已用于开发，不属于未经调试的独立研究测试集。

报告分别统计任务完成、执行成本和迁移增益。原始腕部载荷默认仅作诊断，不设置通用力门槛或加权总分。具体定义和解释范围见[指标说明](docs/metrics.md)。

在本地浏览项目附带的录像：

```sh
python -m http.server 8771 --bind 127.0.0.1 --directory assets
```

打开 `http://127.0.0.1:8771/`。报告包含成功与失败 episode、源任务记录和完整性哈希。

### 接入自己的模型或 agent

使用 `--agent examples/process_agent.py` 接入外部推理进程，通过 `EAH_AGENT_COMMAND` 指定使用显式状态的 UTF-8 JSON worker。接入低层策略时，可在 `skills.execute` 中完成动作坐标系和单位转换。实验者需要自行固定模型版本并记录推理预算，接口细节见[接入说明](docs/integrations.md)。

### 开发检查

```sh
python -m pip install -e ".[dev]"
python -m pip install numpy==1.26.4 imageio imageio-ffmpeg pillow
python -m pytest -q
ruff check src tests scripts examples participant
ruff format --check src tests scripts examples participant
python scripts/check_repository.py
```

[架构说明](docs/architecture.md) · [贡献与发布](CONTRIBUTING.md) · [更新记录](CHANGELOG.md)

### 许可证

项目自身代码使用 MIT 许可证。机器人和环境资产、第三方依赖保留[原有署名与许可证](THIRD_PARTY.md)。
