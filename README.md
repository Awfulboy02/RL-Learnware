# Policy Learnware v0.1

这是 Policy Learnware 的受控 dynamics-shift 诊断分支。它复用冻结的 v0
TaskSpec 表征与 `outer006` policy pool，判断两个先于 TransferSpec 的问题：

1. 同一任务内的受控动力学变化，是否产生值得调度 policy 的候选相关迁移差异；
2. v0 的 candidate-independent TaskSpec，是否能稳定感知该变化。

本仓库只版本化代码、配置和测试。policy bundle、probe 数据、模型参数、数值矩阵和
正式 receipt 属于外部 artifacts，不进入任何 Git 分支。

## 冻结状态与正式结论

首次且唯一的 v0.1 formal run 已完成：

- experiment：`dmc2-damping-outer006-v01-r0`；
- 正式实现提交：`33514ea98e631035c28884e5d8db182a3f93ecad`；
- 正式配置：`configs/dmc2_damping_v01.yaml`；
- config digest：`8966a1f38bd4e4f46c54e6fdb5a8ac004a33fb2eb512187731bced2485ebc2d1`；
- final decision：`NO_GO_CURRENT_POOL_SHIFT`；
- `formal_complete=true`。

| 检查 | 正式结果 | 含义 |
|---|---:|---|
| Gate 0 | PASS | shift、identity、finite rollout、instance isolation、policy parity 与 v0 regression 完整 |
| Gate A | FAIL | 当前 pool/grid 没有同时满足 material、heterogeneity 与 ranking evidence 的 context |
| Gate B | FAIL | TaskSpec 保持 100% source routing，但 shift signal 未越过 bank 内波动与 severity 门槛 |
| Gate C | diagnostic only | 不产生 pass/fail，也不参与最终决策 |
| Gate D | PASS | measurement、oracle 与 private context 的能力边界完整 |
| recomputation audit | PASS | raw shard、aggregation、TaskSpec primitive 与 gate 复算一致 |

Gate A/B 的 `FAIL` 是预注册协议下的有效科学 No-Go，不是工程失败，也不能作为删除
formal artifacts 的理由。当前可以声称：该实验工程闭环和复算均完成，但当前
`outer006` pool 与五点 damping grid 不支持继续声称存在稳定的 policy 调度问题。

关键 receipt SHA-256：

- `completion_manifest.json`：
  `229303fb476d2ef7febbad6f52b277026e4dcfcea81985430f15056a5daa2f32`；
- `analysis/summary.md`：
  `a36500111baa1a7d327232080ac33168bf1faa24327c31cb7df8306ad76108c1`；
- Gate A：
  `70deb95317e28b08af938c36dd03aa910146eeed9b62127b3c977be2bc68f6c4`；
- Gate B：
  `cfa854a128753253e019e252cbcc715a82a077a4912633b310c6c94c0af752b4`；
- Gate D：
  `9bb0bf9d8c012c614c5d8163b673dbf2b17ebda415a6fcc2a7b042c7bbbb117e`；
- recomputation audit：
  `c1a5cf855db30d7bfa76966440eb388bac1c425d490ba01f0fb80a971b15d8a5`。

后续文档维护提交不改变上述历史身份。需要 byte-identical formal revalidation 时，必须
检出正式实现提交，而不是把维护提交重新写进既有 receipt。

## 冻结实验边界

- tasks：`WalkerWalk` 与 `FingerTurnEasy`；
- operator：`global_nonzero_dof_damping_scale`；
- factors：`[0.5, 0.75, 1.0, 1.5, 2.0]`，episode 内静态；
- 每个任务 10 个 candidates：PPO/FPO 各 5 seeds；
- checkpoint：`outer_000006`，5,898,240 environment steps；
- policy 不微调、不续训、不换 checkpoint；
- observation/action schema、reward、reset、termination、goal、horizon、action repeat
  和 morphology 不变；
- shifted candidate rollout 只进入 private oracle，TaskSpec 不读取 factor、candidate、
  algorithm、return 或 bundle 路径；
- 不声称 held-out/OOD、真实 sim-to-real、开放 morphology 或鲁棒 policy selection。

正式覆盖包括 100 份 probe dataset、6,400 episodes、6,400,000 transitions，以及
100 个 oracle shards、5,000 episodes。v0.1 不训练新 selector，也不修改 v0 上传阶段。

## 代码与运行入口

核心实现位于 `src/policy_learnware_v0/v01/`：

- `config.py`、`schemas.py`、`seeds.py`：冻结配置、协议和 seed 合同；
- `variant_env.py`、`probe.py`、`live_binding.py`：shifted 环境与 candidate-independent
  measurement；
- `taskspec.py`、`plans.py`：TaskSpec matrix 和冻结 pair plan；
- `oracle.py`、`statistics.py`、`analysis.py`、`gates.py`：private oracle 与 gates；
- `audit.py`、`recompute.py`、`execution_profile.py`：能力隔离、复算和资源证据；
- `artifacts.py`、`base_runtime.py`、`report.py`、`cli.py`：不可变 artifact、v0 绑定、
  release 和命令编排。

安装后使用 console entry point：

```bash
policy-learnware-v01 --help
policy-learnware-v01 audit-recompute --help
```

无需安装 console script 时，使用同一入口：

```bash
PYTHONPATH=src python -m policy_learnware_v0.v01.cli --help
PYTHONPATH=src python -m policy_learnware_v0.v01.cli audit-recompute --help
```

正式阶段顺序固定为：

```text
validate-config
  -> freeze-run
  -> audit-variants
  -> collect-probes
  -> compute-taskspec-matrix
  -> evaluate-oracle
  -> evaluate-gates
  -> audit-recompute
  -> build-report
```

`audit-recompute` 已由上述 CLI 直接提供，不再维护等价的顶层 wrapper。

## 统一 artifact 根

最终服务器布局使用与两个代码仓并列的统一 `artifacts/`。推荐结构：

```text
RL_Learnware/
  policy_learnware_ope/
  policy_learnware_v0/
  artifacts/
    relocation_manifest.json
    v0/
      runs/
        dmc6-outer006-policy-learnware-v0/
    v01/
      runs/
        dmc2-damping-outer006-v01-r0/
    shared/
      runtime/
        fpo-418c2554/
      repro_fpo_ppo/
        legacy-v02/
          runs/
            full/
  reports/
```

`shared/repro_fpo_ppo/legacy-v02/runs/full/` 只是现存的 60-run 冷备份。该副本缺失原
`_vendor/` runtime，状态必须标为 `incomplete`；它不能被描述为 original runtime，亦不能
单独满足正式复现条件。

root 选择顺序必须是：

1. 命令行显式 `--artifacts-root`、`--base-artifacts-root`、`--measurement-root`、
   `--oracle-root` 等参数；
2. 调用方设置的 `RL_LEARNWARE_ARTIFACTS_ROOT`；
3. 当前 `policy_learnware_v0` 仓库的 sibling `../artifacts`。

CLI 本身只消费显式参数，不会静默读取环境变量。环境变量和 sibling fallback 是 launcher
约定，最终仍应展开为 CLI 参数。以下 shell 片段完成 env → sibling 的解析；在命令中直接
传入其他绝对路径时，显式参数自然优先：

```bash
REPO_ROOT="$(pwd -P)"
PROJECT_ROOT="$(dirname "$REPO_ROOT")"

if [ -n "$RL_LEARNWARE_ARTIFACTS_ROOT" ]; then
  ARTIFACTS_ROOT="$RL_LEARNWARE_ARTIFACTS_ROOT"
else
  ARTIFACTS_ROOT="$PROJECT_ROOT/artifacts"
fi

RELOCATION_MANIFEST="$ARTIFACTS_ROOT/relocation_manifest.json"
V0_BASE="$ARTIFACTS_ROOT/v0/runs"
V01_PARENT="$ARTIFACTS_ROOT/v01/runs"
V01_EXPERIMENT="dmc2-damping-outer006-v01-r0"
V01_RUN="$V01_PARENT/$V01_EXPERIMENT"
POLICY_RUNS="$ARTIFACTS_ROOT/shared/repro_fpo_ppo/legacy-v02/runs/full"
FPO_ROOT="$ARTIFACTS_ROOT/shared/runtime/fpo-418c2554"
```

`V0_BASE` 必须是包含 pool ID 目录的父目录，不能直接指向
`dmc6-outer006-policy-learnware-v0/`。`V01_PARENT` 同理必须是 experiment ID 的父目录。

## Relocation manifest

根权威 manifest 的唯一名称是 `$ARTIFACTS_ROOT/relocation_manifest.json`。它是外部位置
说明，不是新的实验 receipt，也不参与 formal protocol/config digest。它记录旧目录前缀到
新目录前缀的映射，以及每棵已迁移资产树的独立内容清单摘要：

```json
{
  "schema": "rl-learnware.artifact-relocation.v1",
  "mappings": [
    {
      "kind": "directory_prefix",
      "source": "/share/songyf/RL_Learnware/policy_learnware_v0/artifacts_retry_roundoff",
      "target": "v0/runs",
      "content_manifest_sha256": "<sha256>",
      "file_count": 0,
      "total_bytes": 0
    },
    {
      "kind": "directory_prefix",
      "source": "/share/songyf/RL_Learnware/policy_learnware_v0/artifacts_v01",
      "target": "v01/runs",
      "content_manifest_sha256": "<sha256>",
      "file_count": 0,
      "total_bytes": 0
    },
    {
      "kind": "directory_prefix",
      "source": "/share/songyf/RL_Learnware/repro_fpo_ppo/runs",
      "target": "shared/repro_fpo_ppo/legacy-v02/runs/full",
      "completeness": "incomplete",
      "known_missing": ["_vendor/"],
      "content_manifest_sha256": "<sha256>",
      "file_count": 0,
      "total_bytes": 0
    }
  ]
}
```

`target` 相对 manifest 所在的统一 artifact 根解析。消费者应执行规范化后的最长前缀匹配，
拒绝歧义映射、路径逃逸和未声明 symlink；映射后仍按原 receipt 中的 SHA-256、bundle
digest、protocol ID 与 Git commit 验真。没有 manifest-aware loader 时，由 launcher 把
`target` 展开为显式 CLI roots；不得原地重写 artifact JSON。

示例中的 digest、`file_count` 和 `total_bytes` 是待迁移工具填写的占位值；非空资产树
不得以零计数发布 relocation manifest。

迁移完成的判据是源树和目标树的逐文件内容清单、文件数和总字节一致。manifest 只证明
“同一批字节现在在哪里”，不能把残缺本地副本提升为 canonical 资产。

## 冻结 YAML 与 receipt 不可改写

以下文件保留历史字节：

- `configs/dmc6_outer006_v0.yaml`；
- `configs/dmc2_damping_v01.yaml`；
- `configs/v01_smoke.yaml`；
- v0/v01 artifact 内的所有 manifest、protocol、gate、matrix sidecar 和 completion receipt。

历史 YAML 中的 `/share/songyf/RL_Learnware/...` 是冻结 provenance 的一部分，不是新的
canonical locator。不能通过文本替换“修复路径”：这会改变 v0 protocol draft hash、
v01 formal config digest 或下游 manifest digest。新位置只通过显式 CLI roots 和 relocation
manifest 表达。

同样禁止：

- 把 smoke artifact 复制或改名为 formal；
- 修改旧 JSON 后重新计算一个“匹配”的 SHA；
- 在已有 experiment root 中用维护提交生成新 work unit；
- 用本地 metadata-only 副本覆盖服务器 canonical tree；
- 因 Gate A/B 为 FAIL 而删除 formal raw data。

## 复现层级

下表定义的是满足全部输入后的能力边界，不是当前资产清点的完成声明。现有 legacy 60-run
副本缺失原 `_vendor/`，且 FPO runtime candidate 仍须按 manifest 和 commit 验真；因此当前
不得声称 L2 已可完成。

| 层级 | 目标 | 最小输入 | 允许的结论 |
|---|---|---|---|
| L0 报告核读 | 阅读冻结结论 | 顶层 reports 与 artifact 内 summary/receipt | 可复述正式结论，不能声称重算 |
| L1 receipt 验真 | 验证迁移与 digest 链 | 完整 manifest、protocol、gate、matrix sidecar 和 relocation content manifest | 可声称字节与 receipt 未变 |
| L2 frozen revalidation（条件能力） | 重跑 `--resume`、Gate/recompute 与必要 parity | 正式提交 `33514ea`、完整且经验证的原 `_vendor` runtime、完整 v0/v01 artifacts、60 个 outer006 bundles、经 commit/content 验真的 FPO runtime | 仅在全部条件满足后可声称原正式运行可复核；当前未满足 |
| L3 fresh rerun | 重新生成数值实验 | 完整原始数据/训练资产与 GPU runtime | 必须使用新 root/receipt；不能覆盖 v01-r0 |

L2 的正式 runtime 记录为 Python 3.12.13、JAX/JAXlib 0.7.2、Playground 0.0.5、
MuJoCo 3.3.6、NumPy 2.5.1。`$FPO_ROOT` 必须匹配 bundle provenance 与记录的 commit，
并通过 content-manifest 审计；路径名称相同本身不构成 original provenance。

## 冻结 receipt 复验命令

以下命令只适用于完整 canonical assets，并应在正式实现提交和 pinned runtime 中运行：

```bash
v01() {
  PYTHONPATH=src python -m policy_learnware_v0.v01.cli "$@"
}

v01 validate-config \
  --config configs/dmc2_damping_v01.yaml \
  --base-artifacts-root "$V0_BASE"

v01 audit-recompute \
  --base-artifacts-root "$V0_BASE" \
  --frozen-root "$V01_RUN/frozen" \
  --benchmark-private-root "$V01_RUN/benchmark_private" \
  --measurement-root "$V01_RUN/measurement" \
  --oracle-root "$V01_RUN/oracle_private" \
  --analysis-root "$V01_RUN/analysis" \
  --computation-backend jax \
  --resume

v01 build-report \
  --base-artifacts-root "$V0_BASE" \
  --frozen-root "$V01_RUN/frozen" \
  --benchmark-private-root "$V01_RUN/benchmark_private" \
  --measurement-root "$V01_RUN/measurement" \
  --oracle-root "$V01_RUN/oracle_private" \
  --analysis-root "$V01_RUN/analysis" \
  --computation-backend jax \
  --resume
```

若需要重新执行 policy parity 或 oracle，额外显式传入 `--fpo-root "$FPO_ROOT"` 与
`--runs-root "$POLICY_RUNS"`。v01 会按 job ID 寻找 relocation 后的 bundle，并继续要求
bundle digest 与冻结 candidate record 完全一致。当前 `$POLICY_RUNS` 指向的 legacy 冷备份
仅可用于恢复审计；在原 `_vendor/` runtime 未恢复并验真前，不得用它宣称完成 L2。

## 测试门禁

本分支的最小门禁：

```bash
PYTHONPATH=src python -m pytest -q tests/v01
PYTHONPATH=src python -m policy_learnware_v0.v01.cli --help
PYTHONPATH=src python -m policy_learnware_v0.v01.cli audit-recompute --help
```

缺少 JAX 的轻量本地环境可以跳过明确标注的 JAX optimization tests，但不得出现 failure。
正式 Linux/JAX 环境还应完成 Gate 0 所调用的 v0 unit/integration regression，以及至少一个
relocation 后 FPO/PPO bundle 的 golden 和 compiled parity。

## Artifact 能力边界

| 区域 | 内容 | TaskSpec compute 可读 |
|---|---|---:|
| `frozen/` | immutable config、base/protocol bindings、contracts、pair/audit plan | 否 |
| `benchmark_private/` | factor/context、shift manifest、environment instance、Gate 0 | 否 |
| `measurement/` | opaque probe、semantic cache、TaskSpec/routing matrices、execution profile | 是 |
| `oracle_private/` | candidate inventory、episode shards、aggregates | 否 |
| `analysis/` | private joins、gates、recompute audit、summary | 否 |
| experiment root | completion receipt | 否 |

`compute-taskspec-matrix` 不接受 oracle 或 private benchmark root。relocation 只能改变这些
能力域所在的父目录，不能合并能力域或扩大命令可见范围。

## v0 基座

v01 依赖冻结的 v0 pool `dmc6-outer006-policy-learnware-v0`，其 protocol ID 为
`60d7d7ef7e6bac9031f59cb09e5d919f545299184cec2a75a3f86e3d46355633`。
v0 的正式 exact-recurrence 结果是 420/420 retrieval 和 420/420 selected-only deployment。
v01 只读取 v0 已验证的 normalizer、encoder、Gaussian kernel、source RKMEs、public pool
及 20 个相关 candidate records，不重新训练 v0 表征或 policy。
