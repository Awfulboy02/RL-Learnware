# Policy Learnware v0.31：规划、结果报告与设计决议

日期：2026-08-28

状态：development design pivot；不是 formal confirmatory 结论

## 1. 文档权威与变更边界

本文件是 v0.31 **唯一**的规划与结果报告，不再另建同义的 v0.31 报告。它覆盖 v0.3
文档中关于“谁是论文主方法”的旧口径，但不覆盖 v0.3 已冻结的实验协议、算法实现和历史
产物。后续论文叙事、方法展示名和最小补充实验以本文件为准。

本次变更遵守硬性 Occam 边界：

- 不改 RKME、Gaussian MMD、reducer、encoder 或 selector 的数值公式；
- 不重写历史 JSON/CSV，不替换 `B3b`、`A-Env`、`M02/B5` 等冻结 ID；
- 不新增 schema、数据合同、兼容层或大规模测试；
- 只增加论文角色元数据、说明文档与后续最小实验安排。

## 2. 核心设计决议

v0.31 将 **Raw-RKME Learnware** 升格为论文主算子，将原来的 learned
EnvironmentSpec 与 competence 融合设计降为比较变种：

| 冻结 artifact ID | v0.31 角色 | 论文展示名 |
|---|---|---|
| `B3b` | **Primary** | **Raw-RKME Learnware** |
| `A-Env` | Variant | Learned EnvironmentSpec, distance-only |
| `M02/B5` | Variant | Learned EnvironmentSpec + global competence fusion |
| `B3a` | Raw-stat control | Raw transition moments |
| `B4a` / `B4b` | Privileged upper bounds | Target-label rankers |

历史 ID 只用于复算和追溯，不再代表论文中的主次关系。`competence_i` 不再进入主方法的
检索分数；它只保留为 source championization、质量元数据或诊断量。若未来作为近似同分
候选的 tie-break 使用，必须单独报告，不能悄然恢复为主评分项。

新主算子为：

\[
\hat i(q)=\arg\min_i
\operatorname{MMD}\!\left(\widehat\Phi_q^{\mathrm{emp}},
\widetilde\Phi_i^{\mathrm{red}}\right).
\]

被降级的 competence 融合变种为：

\[
s_i(q)=\log c_i-d_i(q)/\sigma.
\]

## 3. 源码事实：Raw-RKME 的输入与计算链

当前 `B3b` 的数值路径不经过 learned encoder，也不存在显式的随机或线性“硬投影矩阵”。
它使用固定 canonicalization，再由 Gaussian kernel 隐式映射到 RKHS：

```text
native (o, a, r, o')
→ source-only z-score normalization
→ global zero-padding + observation/action masks
→ FULL raw packed event
→ Gaussian empirical KME
→ source Reduced RKME / query Empirical KME
→ empirical-to-reduced MMD nearest-neighbour reuse
```

当前冻结宽度为 `D_o=24`、`D_a=6`，FULL packed event 为
`(o, o_mask, a, a_mask, r, o', o'_mask)`，因此每个 transition point 的宽度是

\[
4D_o+2D_a+1=4\times24+2\times6+1=109.
\]

`terminated/truncated` 用于 episode 边界与 episode-balanced weighting，不进入 109 维点。
现有 fused development runner 会同时构建 Raw 与 R5 路径，因此工程入口仍会加载 R5
checkpoint；但 `B3b` 的点、kernel、RKME、距离和分数均不读取 R5 输出。将 Raw-only runner
从这项非数值依赖中拆出是可选的后续工程降本，不是本次角色调整或结果成立的前提。

## 4. v0.3 development 结果

结果范围为 30 个 source exact-recurrence contexts 加 24 个 development interpolation
contexts，共 54 个 contexts。该运行是 `formal=false`，未读取 confirmatory 或
extrapolation oracle。

| 方法 | Source regret ↓ | Dev regret ↓ | All regret ↓ | Task-compatible ↑ | Exact anchor ↑ | Top-3 oracle ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `B3a` Raw moments | 0.0800 | 0.1078 | **0.0923** | 100.0% | 66.7% | 74.1% |
| `B3b` **Raw-RKME** | **0.0806** | 0.1334 | **0.1040** | 100.0% | **43.3%** | 63.0% |
| `A-Env` distance-only | 0.1345 | **0.0995** | 0.1190 | 100.0% | 16.7% | **74.1%** |
| `M02/B5` encoder + competence | 0.1728 | 0.1686 | 0.1709 | 83.3% | 16.7% | 59.3% |
| `B4a` privileged kNN | 0.0662 | 0.0424 | 0.0557 | 98.1% | 16.7% | 81.5% |
| `B4b` privileged ridge | 0.0423 | 0.0091 | 0.0276 | 100.0% | 33.3% | 87.0% |

主要结论：

1. Raw-RKME 相对 `M02/B5` 将总体 regret 从 `0.1709` 降至 `0.1040`，相对改善
   **39.1%**，并将 task-compatible selection 从 `83.3%` 提高到 `100%`。
2. 在排除 raw-coordinate moments (`B3a`) 与读取 target policy-return labels 的
   privileged upper bounds (`B4a/B4b`) 后，Raw-RKME 是本轮 pooled 结果最强的方法。
3. Raw-RKME 并非所有 regime 全面支配：`A-Env` 在 development interpolation 上的
   regret 为 `0.0995`，优于 Raw-RKME 的 `0.1334`，Top-3 coverage 也更高。因此
   `A-Env` 应保留为重要的互补比较变种。
4. `A-Env` 与 `M02/B5` 使用同一 R5 表示。只加入未经跨任务校准的 global competence
   后，总体 regret 从 `0.1190` 恶化到 `0.1709`，相容率从 `100%` 降到 `83.3%`；9 个
   不相容选择全部来自 CartpoleSwingup。当前证据否定的是这项融合方式，而不是
   competence 作为 source-side 质量控制的所有用途。

`B3a` 仍是所有无 target-label 方法中的最低总体 regret，但它公开 raw-coordinate 低阶
统计，且不是 RKME 学件接口，所以保留为强 control，而不取代论文主算子。

## 5. Transition Signal Atlas 对设计决议的支持

当前 dynamics-facing panel 包含 22 个数值 cells / 50 个 seed-level works，不是完整
39-logical-cell 14-control Atlas。最强的经验 dynamics 规约来自 Raw Delta + action：

| Transition 规约 | Representation | Task/ABI Top-1 | Axis bracket@2 | ρ(log-factor) | Repeat separation | P(ratio>1) |
|---|---|---:|---:|---:|---:|---:|
| FULL `(o,masks,a,r,o')` | Raw KME | 1.000 | 0.542 | 0.229 | 1.225 | 0.667 |
| Reward-free `(o,a,o')` | Raw KME | 0.958 | 0.583 | 0.208 | 1.518 | 0.700 |
| **Delta + action `(o'-o,a)`** | **Raw KME** | **1.000** | **0.750** | **0.417** | **2.206** | **0.967** |
| Delta + action `(o'-o,a)` | R5 CORRO-style MLP | 1.000 | 0.569 | 0.278 | 1.608 | 0.833 |

这说明简单 Raw geometry 已包含很强的异构 task/ABI 路由信号，并且 Delta + action 对同
task 内 dynamics axis/factor 的规约最强。当前 task-SupCon R5 训练会把同 task 的多个
dynamics anchors 拉近，可能压缩了细粒度 dynamics geometry。

但必须分开两类证据：已完成的 end-to-end `B3b` 使用 FULL packed event；最强
dynamics 结果来自 `(o'-o,a)`。现有结果不能写成“Delta Raw-RKME 已在策略选择端获胜”。
若论文需要 dynamics-only 主张，必须补一项最小的 view-specific end-to-end 对照。

## 6. 与学件理论及隐私口径的关系

Raw-RKME 更直接对应“提供者发布 reduced specification，复用者持有 query empirical
specification 并做非交互匹配”的学件接口。它不要求复用者读取 source raw transitions，
也不要求上传者读取 query raw transitions。这里的核心性质是 data inaccessibility 与
non-interaction，不是 differential privacy。

当前 development runner 是集中式实验实现：它会读取 source banks 来拟合 source-only
normalizer/bandwidth，也会在同一运行根生成 query empirical points。因此，本轮只验证数值
接口，不等同于已经实现了提供者/复用者的物理隔离部署。此外，ReducedRKME supports 仍在
可解释的 raw transition 坐标中，可能暴露代表性原型；本项目没有证明不可逆、零泄漏或
差分隐私。

几类方法的风险必须区分：

- `B3a` 发布 raw-coordinate 低阶统计，具有更直接的统计暴露风险，但本轮没有证明样本重构；
- `B4a/B4b` 读取 development policy-return labels，属于 privileged target supervision，
  不是 source raw-data 泄漏；
- competence 是 source performance/business metadata，不是 dynamics identity，也不是原始
  transition 样本。移除它可减少元数据依赖，但不会自动建立形式化隐私。

严格部署时应让 source reduction 在提供者侧完成、query empirical KME 与匹配在复用者侧
本地完成，并且不公开 query points。本轮不为此新增隐私合同或测试；先如实披露边界。

## 7. 论文贡献的重新定位

Raw RKME/Gaussian MMD 是继承的数学工具，不能把“发明 RKME”作为 novelty。v0.31 将论文
贡献集中到：

1. 面向 RL 异构 embodiment、goal 与 dynamics 的 transition measurement 规约；
2. candidate-independent probe、global canonicalization 与跨 ABI 可比较性；
3. 14-control Signal Atlas 对 schema、reward、occupancy、transition pairing 与 dynamics
   信号的可识别边界分析；
4. source Reduced RKME / query Empirical KME 的非交互策略学件复用实例，以及从规约距离到
   冻结 policy ranking/regret 的闭环验证。

learned encoder 不再是 v0.31 的主创新。当前 R5 与 competence 结果作为“为何更复杂设计
未必更好”的机制比较；更广泛的 encoder-family 研究仍属于 v0.4。

## 8. v0.31 最小推进计划

按优先级只保留以下工作：

1. **角色与命名冻结（本次完成）**：保留 artifact ID，更新代码中的 paper-role metadata、
   README、v0.3 superseding notice 与论文框架任务。
2. **最小 view 对照**：复用现有 banks 与 selector，比较 FULL、reward-free `(o,a,o')`、
   Delta + action `(o'-o,a)` 三个 Raw-RKME end-to-end policy ranking；不重训 policy/encoder。
3. **confirmatory/extrapolation**：在新主方法上执行预先冻结的 confirmatory 与
   extrapolation 评估，分开报告 source/dev/extrapolation，不只报告 pooled mean。
4. **可选工程降本**：只有当独立复现需要时，才把 Raw-only runner 从 R5 checkpoint 的
   非数值依赖中拆出；不得借机增加 schema、合同或全量测试。
5. **论文更新**：方法、贡献与摘要以 Raw-RKME 为主，保留 `A-Env` 的 interpolation 优势、
   FULL reward/mask shortcut、集中式 runner 与 reduced-support inversion 风险。

### 明确不做

- 不把 frozen `B3b` 改名为新的 artifact ID，也不伪造新的历史结果；
- 不因本次角色变化重跑 policy、encoder、RKME reduction 或已完成 baselines；
- 不恢复已剃除的审计/合同，不新增大规模单元测试；
- 不把 development 结果包装为 formal superiority；
- 不把当前 CORRO-style 失败外推为“所有 learned encoder 均无价值”。

## 9. 允许与禁止的当前论文表述

当前允许：

> 在本轮 development panel 中，canonical Raw-RKME 在无 target policy-return labels 的
> RKME 方法中取得最低 pooled regret，并显著优于当前 learned EnvironmentSpec + global
> competence 融合；Raw Delta + action 同时给出最强的同任务 dynamics 规约信号。

当前禁止：

- “Raw-RKME 已在 confirmatory/extrapolation 上显著优于所有方法”；
- “Raw-RKME 实现差分隐私、不可逆或绝对零泄漏”；
- “Delta Raw-RKME 已在 end-to-end policy selection 上获胜”；
- “task、embodiment 与 ABI 已被独立识别”（当前 panel 中三者一一对应）；
- “learned encoder 普遍无效”或“competence 在任何用途中都无效”。

## 10. 本次变更记录与完成条件

- [x] 冻结 `B3b` 为历史 artifact ID，并登记为 v0.31 论文主方法；
- [x] 将 `A-Env` 与 `M02/B5` 登记为比较变种；
- [x] 更新 README 与 anonymous-market runtime note；
- [x] 建立本文件作为唯一 v0.31 规划/报告；
- [x] 在根目录 v0.3 规划末尾追加醒目的 superseding notice；
- [x] 向任务“RL论文框架搭建”发送本轮结果和设计决议；
- [x] 完成轻量一致性检查；本提交随后推送 `v03` 线上分支。

完成这些文档/元数据动作后，v0.31 的设计转向即落盘；后续最小 view 对照和
confirmatory/extrapolation 属于实验推进，不阻断本次命名调整。
