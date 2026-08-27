# Policy Learnware v0.31：规划、结果报告与设计决议

日期：2026-08-28

状态：development design pivot + fixed-Raw-RKME view 对照完成；不是 formal confirmatory 结论

## 1. 文档权威与变更边界

本文件是 v0.31 **唯一**的规划与结果报告，不再另建同义的 v0.31 报告。它覆盖 v0.3
文档中关于“谁是论文主方法”的旧口径，但不覆盖 v0.3 已冻结的实验协议、算法实现和历史
产物。后续论文叙事、方法展示名和最小补充实验以本文件为准。

本次变更遵守硬性 Occam 边界：

- 不改 RKME、Gaussian MMD、reducer、encoder 或 selector 的数值公式；
- 不重写历史 JSON/CSV，不替换 `B3b`、`A-Env`、`M02/B5` 等冻结 ID；
- 不新增通用数据合同、兼容层或大规模测试；
- 只增加一个独立 Raw-only runner、其最小结果格式与论文表格，不修改冻结 Atlas。

## 2. 核心设计决议

v0.31 将 **Raw-RKME Learnware** 升格为论文主算子族，将原来的 learned
EnvironmentSpec 与 competence 融合设计降为比较变种。补充 view 对照进一步选出
`Delta + action` 作为 development 阶段的主 measurement 候选：

| artifact / candidate | v0.31 角色 | 论文展示名 |
|---|---|---|
| v0.31 dev candidate（不伪造冻结 ID） | **Primary candidate** | **Raw-Delta RKME Learnware** |
| `B3b` | Historical FULL comparator | Raw-FULL RKME Learnware |
| `A-Env` | Variant | Learned EnvironmentSpec, distance-only |
| `M02/B5` | Variant | Learned EnvironmentSpec + global competence fusion |
| `B3a` | Raw-stat control | Raw transition moments |
| `B4a` / `B4b` | Privileged upper bounds | Target-label rankers |

历史 ID 只用于复算和追溯，不再代表论文中的主次关系。`competence_i` 不再进入主方法的
检索分数；它只保留为 source championization、质量元数据或诊断量。若未来作为近似同分
候选的 tie-break 使用，必须单独报告，不能悄然恢复为主评分项。

主算子对 transition measurement `v` 的统一形式为：

\[
\hat i(q)=\arg\min_i
\operatorname{MMD}\!\left(\widehat\Phi_{q,v}^{\mathrm{emp}},
\widetilde\Phi_{i,v}^{\mathrm{red}}\right),
\qquad \phi_{\Delta a}(\tau)=(o'-o,a).
\]

这里选中 `Delta + action` 使用了同一六任务 development panel，因此它是需要在未读取的
confirmatory/extrapolation 上验证的候选，而不是 method-selection-blind formal 结论。

被降级的 competence 融合变种为：

\[
s_i(q)=\log c_i-d_i(q)/\sigma.
\]

## 3. 源码事实：Raw-RKME 的输入与计算链

冻结 `B3b` 的数值路径不经过 learned encoder，也不存在显式的随机或线性“硬投影矩阵”。
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
v0.31 已用独立 Raw-only runner 去除这项工程上的 R5 checkpoint 依赖。新候选只把 FULL
measurement 替换为 `(o'-o,a)`；canonicalizer、episode-balanced weighting、Gaussian kernel、
source Reduced RKME、query Empirical KME、support budget 与 MMD ranking 均保持不变。FULL
重算带宽逐值复现冻结 `B3b` 的 `8.918560970286762`，all regret 也复现为 `0.1040333914`。

## 4. v0.3 development 结果

结果范围为 30 个 source exact-recurrence contexts 加 24 个 development interpolation
contexts，共 54 个 contexts。该运行是 `formal=false`，未读取 confirmatory 或
extrapolation oracle。

| 方法 | Source regret ↓ | Dev regret ↓ | All regret ↓ | Task-compatible ↑ | Exact anchor ↑ | Top-3 oracle ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `B3a` Raw moments | 0.0800 | 0.1078 | 0.0923 | 100.0% | 66.7% | 74.1% |
| `B3b` Raw-RKME | 0.0806 | 0.1334 | 0.1040 | 100.0% | 43.3% | 63.0% |
| `A-Env` distance-only | 0.1345 | 0.0995 | 0.1190 | 100.0% | 16.7% | 74.1% |
| `M02/B5` encoder + competence | 0.1728 | 0.1686 | 0.1709 | 83.3% | 16.7% | 59.3% |
| `B4a` privileged kNN | 0.0662 | 0.0424 | 0.0557 | 98.1% | 16.7% | 81.5% |
| `B4b` privileged ridge | 0.0423 | 0.0091 | 0.0276 | 100.0% | 33.3% | 87.0% |

主要结论：

1. FULL Raw-RKME 相对 `M02/B5` 将总体 regret 从 `0.1709` 降至 `0.1040`，相对改善
   **39.1%**，并将 task-compatible selection 从 `83.3%` 提高到 `100%`。
2. 在该历史 FULL measurement 下，排除 raw-coordinate moments (`B3a`) 与读取 target
   policy-return labels 的 privileged upper bounds (`B4a/B4b`) 后，Raw-RKME 是 pooled
   结果最强的方法；第 5 节的 fixed-operator 对照随后改写了 view 选择结论。
3. `A-Env` 在 development interpolation 上的 regret 为 `0.0995`，优于 FULL Raw-RKME
   的 `0.1334`；但新 Raw-Delta 候选进一步降至 `0.0699`。`A-Env` 仍作为 learned
   representation 比较变种保留，而不再是 development interpolation 最优无标签方法。
4. `A-Env` 与 `M02/B5` 使用同一 R5 表示。只加入未经跨任务校准的 global competence
   后，总体 regret 从 `0.1190` 恶化到 `0.1709`，相容率从 `100%` 降到 `83.3%`；9 个
   不相容选择全部来自 CartpoleSwingup。当前证据否定的是这项融合方式，而不是
   competence 作为 source-side 质量控制的所有用途。

`B3a` 在这张历史 baseline 表中是最低总体 regret 的无 target-label 方法；新 Raw-Delta
RKME 的总体 regret 为 `0.0413`，已经低于 `B3a` 的 `0.0923`。`B3a` 仍因直接发布
raw-coordinate 低阶统计且不属于 RKME 学件接口而保留为强 control。

## 5. 重置版 Table 1：固定 Raw-RKME 的 transition controls

本表不再混合 representation family。八行全部固定为
`R0 raw identity → source Reduced RKME / query Empirical KME → nearest Gaussian MMD`，只改变
transition measurement。它复用同一批 30 source、24 development banks 与已经完成的
54-context policy oracle，不重训 policy/encoder，也未读取 confirmatory/extrapolation。
每个 view 都按同一 source-only calibration 规则独立估计 bandwidth；固定的是算法协议，
不是跨 view 强行共用一个带宽数值。`C_RF_SHUFFLED_NEXT` 的 84 个 measured banks 均通过
observation/action/next-state marginal preservation 与 pairing-destroyed audit。

### 5.1 Panel A：绝对结果

| View | Channels | Role | Task/ABI@1 (n=24) | Repeat exact-source@1 (n=30) | Axis bracket@2 | ρ(log-factor) | Repeat separation | Source regret | Dev regret | All regret | Task-compatible |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | `(o,masks,a,r,o')` | historical paper operator | 1.000 | 0.433 | 0.542 | 0.229 | 1.726 | 0.0806 | 0.1334 | 0.1040 | 1.000 |
| No mask | `(o,a,r,o')` | mask control | 0.958 | 0.433 | 0.583 | 0.250 | 1.754 | 0.1299 | 0.1626 | 0.1445 | 0.944 |
| Reward-free | `(o,a,o')` | reward control | 0.958 | 0.467 | 0.583 | 0.208 | 1.835 | 0.1265 | 0.1577 | 0.1404 | 0.944 |
| RF shuffled-next | `(o,a,perm(o'))` | pairing control | 0.917 | 0.467 | 0.542 | 0.167 | 1.833 | 0.1660 | 0.1934 | 0.1782 | 0.907 |
| **Delta + action** | **`(o'-o,a)`** | **development primary candidate** | **1.000** | **0.633** | **0.750** | **0.417** | **2.250** | **0.0185** | **0.0699** | **0.0413** | **1.000** |
| Action values only | `(a)` | clean Delta reference | 0.792 | 0.067 | 0.250 | -0.146 | 1.169 | 0.4994 | 0.2971 | 0.4095 | 0.630 |
| State + action | `(o,a)` | next-state control | 0.958 | 0.500 | 0.583 | 0.188 | 1.725 | 0.1308 | 0.1577 | 0.1427 | 0.944 |
| State only | `(o)` | occupancy/action control | 0.958 | 0.533 | 0.583 | 0.188 | 1.996 | 0.1038 | 0.1577 | 0.1277 | 0.963 |

### 5.2 Panel B：严格单变量对照

所有 benefit 均统一成“正值有利于 richer view”；W/T/L 是 richer 相对 control 的逐 context
regret 胜/平/负。由于大量 contexts 的 top policy 不变，均值与 W/T/L 必须同时报告。

| 因素 | Richer view | Control | Dev regret benefit | All regret benefit | Axis benefit | Repeat benefit | Dev W/T/L | All W/T/L |
|---|---|---|---:|---:|---:|---:|---:|---:|
| masks | FULL | No mask | 0.0293 | 0.0404 | -0.0417 | -0.0274 | 1/22/1 | 3/49/2 |
| reward channel | No mask | Reward-free | -0.0050 | -0.0041 | 0.0000 | -0.0818 | 1/21/2 | 1/50/3 |
| next-state pairing | Reward-free | RF shuffled-next | 0.0358 | 0.0378 | 0.0417 | 0.0023 | 1/23/0 | 4/50/0 |
| next-state channel | Reward-free | State + action | 0.0000 | 0.0024 | 0.0000 | 0.1102 | 0/24/0 | 1/53/0 |
| state delta | Delta + action | Action only | **0.2272** | **0.3682** | **0.5000** | **1.0811** | **13/7/4** | **37/12/5** |
| action channel | State + action | State only | 0.0000 | -0.0150 | 0.0000 | -0.2705 | 0/24/0 | 0/53/1 |

### 5.3 结果解释与旧表边界

1. **Delta 是本轮同时强化 signal geometry 与 policy reuse 的最大幅度因素。** 相对 action-only，
   它把 axis bracket@2 提高 `0.5000`、repeat ratio 提高 `1.0811`、dev regret 降低
   `0.2272`。其 all regret `0.0413` 相对 FULL 的 `0.1040` 改善约 **60.3%**，也低于
   `B3a` (`0.0923`)、`A-Env` (`0.1190`) 和 privileged kNN `B4a` (`0.0557`)；但仍未超过
   privileged ridge `B4b` (`0.0276`)。这些是同一 development panel 上的描述性比较；由于
   Delta view 也是在该 panel 上选出的，不能外推为胜过 privileged upper bound 的泛化结论。
2. **coupling 比 next-state marginal presence 更关键。** 保持 `(o,a,o')` 同通道、只置乱
   `o'` 后，dev/all regret 分别恶化 `0.0358/0.0378`；单纯删除 `o'` 则 dev regret 不变。
   不过 pairing 的 W/T/L 为 `4/50/0`（all），说明效应集中在少数 contexts，不能包装成普遍改善。
3. **mask 结果更符合跨 ABI/schema 路由，而不是更好的细粒度 dynamics 几何。** FULL 改善 all
   regret `0.0404`，但 axis bracket 与 repeat separation 反而下降，而且 W/T/L 仅为
   `3/49/2`；task/embodiment/ABI 在当前 panel 中又一一对应，因此这只是 consistent-with
   evidence，不能写成独立因果识别。
4. **reward 与 action marginal 未观察到稳健正收益。** reward channel 的 dev/all benefit 均为负；
   `(o,a)` 相对 `(o)` 在 development 上完全持平。当前 evidence 更支持 state-change/coupling，
   而不是 reward 或动作边际本身。

旧 `transition_semantic_contract_comparison.{md,csv}` 与 22-cell/50-work Atlas 结果原样保留为
legacy backup。旧表使用 Signal runner 的另一套 bandwidth/reduction protocol，不能与本表逐行
拼数值；它只作为趋势交叉验证：旧结果同样把 Raw Delta + action 判为最强 dynamics 规约，并
显示 task-SupCon R5 可能压缩同任务内 dynamics geometry。

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
2. **最小 view 对照（已完成）**：统一重算八个 fixed-Raw-RKME views，补齐 no-mask、
   reward-free shuffled-next 与 clean action-only controls，并复用既有 oracle 完成 ranking。
3. **confirmatory/extrapolation（下一步）**：冻结 Raw-Delta 为 primary candidate、FULL 为
   historical comparator，在未读取的 confirmatory 与 extrapolation 上评估，分开报告
   source/dev/extrapolation，不只报告 pooled mean。
4. **Raw-only 工程降本（已完成）**：独立 runner 不加载 R5 checkpoint，不采新轨迹、不跑
   policy/encoder；不得借机恢复 schema、合同或全量测试。
5. **论文更新**：方法、贡献与摘要以 Raw-Delta RKME candidate 为主，FULL 作为 frozen
   historical comparator；保留集中式 runner、development method selection 与
   reduced-support inversion 风险。

### 明确不做

- 不把 frozen `B3b` 改名为新的 artifact ID，也不伪造新的历史结果；
- 不重跑 policy、encoder 或历史 baselines；只执行重置 Table 1 必需的八组 Raw-RKME reductions；
- 不恢复已剃除的审计/合同，不新增大规模单元测试；
- 不把 development 结果包装为 formal superiority；
- 不读取 confirmatory/extrapolation 来继续选择 transition view；
- 不把当前 CORRO-style 失败外推为“所有 learned encoder 均无价值”。

## 9. 允许与禁止的当前论文表述

当前允许：

> 在本轮 development panel 中，固定 Raw-RKME 后，Delta + action measurement 同时取得
> 最强的同任务 dynamics 规约，并使 per-query 不读取 target policy-return labels 的 scorer
> 取得最低 regret；其数值低于历史 FULL Raw-RKME 及当前 learned EnvironmentSpec + global
> competence 融合。该 view 使用 development oracle 选出，仍需在未读取的
> confirmatory/extrapolation 上验证。

当前禁止：

- “Raw-RKME 已在 confirmatory/extrapolation 上显著优于所有方法”；
- “Raw-RKME 实现差分隐私、不可逆或绝对零泄漏”；
- “Raw-Delta 的 view 是 method-selection-blind 预注册结果”或“已经 formal 获胜”；
- “mask 改善证明了内在 dynamics 识别”（当前更符合 schema/ABI routing）；
- “task、embodiment 与 ABI 已被独立识别”（当前 panel 中三者一一对应）；
- “learned encoder 普遍无效”或“competence 在任何用途中都无效”。

## 10. 本次变更记录与完成条件

- [x] 冻结 `B3b` 为历史 artifact ID；完成 view 对照后将其登记为 FULL comparator；
- [x] 将 `A-Env` 与 `M02/B5` 登记为比较变种；
- [x] 更新 README 与 anonymous-market runtime note；
- [x] 建立本文件作为唯一 v0.31 规划/报告；
- [x] 在根目录 v0.3 规划末尾追加醒目的 superseding notice；
- [x] 向任务“RL论文框架搭建”发送本轮结果和设计决议；
- [x] 完成轻量一致性检查；
- [x] 新增独立 Raw-only runner，不加载 R5、policy runtime 或 confirmatory 资产；
- [x] 完成八个统一 production Raw-RKME view 对照及 30/24/54 覆盖验收；
- [x] 生成重置版 Table 1 Panel A/B，旧表与旧运行原样保留；
- [x] 将 Raw-Delta 登记为下一阶段 primary candidate，FULL 保留为历史 comparator。

权威 development 运行根：
`/share/songyf/RL_Learnware/v03_main_runs/v031-raw-transition-controls-20260828-r1`。
本地论文表与汇总位于 `v03_results_analysis/raw_rkme_transition_contract_table_v031.{md,csv}`
及 `v03_results_analysis/raw/v031_table1_summary.json`；旧表文件没有删除或覆盖。

至此 v0.31 development 设计转向与最小 view 对照均已落盘。下一项数值工作是冻结当前
candidate 后执行 confirmatory/extrapolation；不得再用该 oracle 继续调 view。
