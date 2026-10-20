## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-10-18（America/Los_Angeles）
- Verification Status：ANALYZED
- Version Label：opsd_zero_credit_quality_review_v1
- 审阅对象：旧固定窗口 OPSD / FA-BM-VAE / SPQC-550，v7 采样准备材料。
- 范围：已有文件核对、已有统计复算、本地回归测试；不是新的独立采样实验，也不是真机质量验证。
- 编制：Codex 辅助技术审阅；人工批准人：未填写；批准状态：待审，不构成提交授权。

# OPSD 真机采样质量规则零额度审阅报告

## 一、结论摘要

**当前结论：本地工程保护检查通过；四项质量阈值仍是待审的经验候选值，尚不足以宣称已验证的联合接受率或真机采样可靠性。**

本次没有调用 Kaiwu 真机接口、消耗 Sampling Credits、重训模型、生成新的 SA 参考样本或计算新的 TEST 预测指标。未修改冻结协议、阈值批准状态、模型、矩阵或历史实验结果。

| 审阅项目 | 证据与结论 | 状态 |
|---|---|---|
| 本地软件保护 | 本次重跑指定回归测试，62 项通过、14 条依赖弃用警告 | 通过限定范围的工程检查 |
| 参考记录规模 | 300 个已有 float-SA 样本文件，各100×48；记录的300个种子互不重复 | 格式/记录检查通过，不等于证明所有统计独立性 |
| 汇总与分位数计算 | 150个“条件×重复对”记录重新汇总成15批；四项q95与报告一致，绝对差小于1e-14 | 已复算 |
| 本地量化对照 | 三个 gain 的已有均值均落入候选上限 | 仅限现有本地对照 |
| 阈值稳定性 | 能量和有效逆温度上限对批次组成仍有可见敏感性 | 谨慎解释 |
| 联合通过率 | 阈值估计数据内为11/15；不是独立验证 | 不能声称95%联合保证 |
| 独立正常对照 | **旧结论，已被后续 v3/v4 审计替代**；v3 提供了 300 个不与历史参考及 v1/v2 交集的正常 SA 样本，v4 为同组复现 | 仍只有一次独立证据，不得合并计数 |
| 分布异常识别能力 | 现有格式/哈希故障测试不能替代分布劣化对照 | 证据缺口 |
| 真机效果、路由与扣费 | 未进行真实采样；本地profile不能证明平台实际路由 | 未验证 |

本报告不是全仓库无缺陷证明，也不是论文录用、量子优势或预测精度提升的保证。

## 二、“批准报告”究竟批准什么

这里的批准是课题内部对实验规则和风险的确认，不是重新向玻色公司申请额度，也不等同于已核验的平台官方额外审批要求。它是本项目为有限额度和研究可追溯性设置的控制步骤。

批准的内容是：

1. 接受指定的经典采样参考、指标定义、聚合单位、阈值及其已知限制。
2. 同意在看到真机结果前冻结规则，不为使结果通过而事后放宽阈值。
3. 同意停止条件：响应身份/格式不合格立即停批；全部候选 gain 质量不合格则不进入 TEST。
4. 确认正式运行应使用哪个协议和准备包，以及谁有权批准继续消耗额度。

批准不代表：真机一定成功、每个条件均合格、全部100 reads统计独立、风电场景指标一定改善，或论文结论已经成立。

| 确认事项 | 它允许什么 | 它不能替代什么 |
|---|---|---|
| 质量规则确认 | 固定接受/停止标准 | 单次真机提交授权、平台许可 |
| 首次 canary 确认 | 一个指定VS条件、gain=125、100 reads的工程测试 | 批量VS批准、TEST解锁 |
| 批量继续确认 | 在条件满足后执行规定范围的剩余任务 | VS质量验收、真实gain冻结 |

人工确认可采用一段明确文字或实验记录，不必为了此步骤额外建立公司级签章流程。代码中的布尔标志不是可信的人身签名；须另保留负责人确认依据。

## 三、输入材料与范围

工作目录：`D:\研究生作业\组会4\generative-models-main`。

主要材料（相对于仓库根目录）：

- 协议：`configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json`
- 准备清单：`export/opsd_sa_negative_kaiwu_preparation_v7/preparation_manifest.json`
- 参考报告：`export/opsd_sa_negative_kaiwu_preparation_v7/quality_gate_reference_v7_revalidated/quality_gate_reference_audit.json`
- 条件级重复对记录：同目录 `quality_gate_null_rows.json`
- float-SA原始文件：同目录 `samples/gain_*/repeat_*/*.npz`
- 最新既有离线mock链路：`quality_gate_reference_v7_revalidated_refresh3/`；该目录中的mock不能作为真实硬件记录。
- 流程要求：[首次采样前收尾清单](OPSD_SPQC550_PRE_SUBMISSION_CHECKLIST.md)。

当前 Git HEAD 为 `f87aa263b92ce12cfde9a5dc810ba486abe4fb00`，但工作树有未提交修改，因此不能仅用该commit代表本次受审代码。主要文件快照哈希见附录。

预算仍为：10个VS条件×3个gain×100 reads＝3,000；90个TEST日期×100 reads＝9,000；计划合计12,000 reads。额度单位与实际扣费关系尚需平台确认。首个100 reads只有在身份/协议兼容且验收合格时才可计入正式VS，不能默认任何失败调用均不扣费。

## 四、规则的实际含义

### 4.1 比较的是什么

用本地经典SA作为固定操作参考，比较采样分布的若干统计特征。不是验证SA为精确玻尔兹曼真值，也不是用最低能量证明概率分布正确。

当前参考配置记录为：每条件100 reads，100 sweeps，float逻辑SA目标beta=1；本地量化矩阵对照使用相应的gain缩放。只适用于相应模型、条件面板、缩放和指标定义；模型或能量尺度改变后不能直接移植阈值。

### 4.2 指标与候选值

先在每个VS条件上计算误差，再在相同10条件内等权平均，得到每个gain的统计量。以下上限作用于这个平均值，不是逐自旋、逐read或逐日期的硬阈值。

| 指标 | 当前实现含义 | 候选上限（显示四舍五入） |
|---|---|---:|
| Energy Wasserstein | 在原始逻辑h,J下计算能量，再比较两组经验能量分位数；代码采用1001个分位点近似 | 0.307866605 |
| Effective-beta error | 候选样本伪似然估计的beta_eff与目标1的绝对偏差；不是两个估计beta之差 | 0.088987960 |
| Edge-moment MAE | J非零边上的自旋乘积均值与参考的平均绝对差；不是中心化相关系数 | 0.115630233 |
| Magnetization MAE | 各逻辑自旋样本均值与参考的平均绝对差 | 0.117079167 |

四项均不超过候选上限，才满足当前候选联合规则。唯一状态比例仅作诊断，没有正式最低值。重复状态本身不是无效数据。

均值达标不能保证每个条件都可靠。条件级异常应保留并报告，不能用总体均值掩盖。若要增加额外的条件级拒绝规则，必须另行明确并在正式结果前冻结，而不是看完结果后添加。

## 五、本次复算结果

### 5.1 参考规模与计算核对

10个条件，每个gain标签下10次独立种子运行；重复按(0,1)、(2,3)、(4,5)、(6,7)、(8,9)组成不共享样本的配对。每对先汇总10条件；三个标签共15个批次。

本次检查了300个已有float-SA文件的100×48形状和±1取值，读取种子记录后未发现重复。检查的是记录与格式，不是完整重跑随机模拟。

本次从150行已有条件级统计重新构造15行均值，并重算四项95%经验分位数，结果与保存的报告一致。本轮未逐一从原始自旋重算全部能量/伪似然指标，未生成新样本。

15批才是这次q95估计所用的批次数，不能把300个文件、30,000条自旋样本或150个条件级记录当成15批估计的独立重复数。

### 5.2 顺序累计的敏感性

| 指标 | 前10批q95 | 全15批q95 | 相对变化 |
|---|---:|---:|---:|
| Energy Wasserstein | 0.266933834 | 0.307866605 | +15.33% |
| Effective-beta error | 0.080142809 | 0.088987960 | +11.04% |
| Edge-moment MAE | 0.116195349 | 0.115630233 | −0.49% |
| Magnetization MAE | 0.117612500 | 0.117079167 | −0.45% |

原表按gain标签分组排列，前10批与后加入5批的组构成不同。因此该变化只能说明顺序/组构成敏感性，不能直接解释为收敛失败或成功，也不能证明某个硬件gain导致了这种差异。

### 5.3 留一批敏感性

每次去掉15批中的一批，用剩余14批重新计算q95，共15次。结果如下：

| 指标 | 留一后q95最小值 | 留一后q95最大值 |
|---|---:|---:|
| Energy Wasserstein | 0.281052761 | 0.309488935 |
| Effective-beta error | 0.080535452 | 0.089017726 |
| Edge-moment MAE | 0.114647674 | 0.115766279 |
| Magnetization MAE | 0.116389583 | 0.117189583 |

这是已有数据的敏感性分析，不是置信区间，不是独立留出验证。不能用“留一计算了15次”增加独立样本量。

### 5.4 四项同时通过的情况

使用全15批估计出来的四项上限，再评价这同15批：**11/15同时达标，约73.33%**。

这一结果只说明当前四项单独q95组成的联合规则，在阈值估计数据上的表现；不能解释为真机成功率73.33%、未来正常样本拒绝率26.67%，或给它套用独立验证置信区间。

四项各自取95%分位数，不会自动产生95%的联合通过保证。正式接受/停止方案是否愿意承担相应误拒风险，仍须通过独立对照评估并由负责人决定。

### 5.5 现有本地量化对照

| 本地int8-SA gain | Energy W | beta误差 | Edge-moment MAE | Magnetization MAE | 当前候选界限内 |
|---|---:|---:|---:|---:|---|
| 125 | 0.187386434 | 0.067692308 | 0.110418605 | 0.106625000 | 四项均在界限内 |
| 150 | 0.217937237 | 0.047230769 | 0.111767442 | 0.110583333 | 四项均在界限内 |
| 200 | 0.183779269 | 0.045565217 | 0.111813953 | 0.111208333 | 四项均在界限内 |

数据来自既有报告，本轮没有重新生成这些量化样本。它们不是SPQC-550输出，不构成真机gain选择，也不能说明硬件上的哪个gain最佳。当前对照使用的float参考与部分阈值估计记录重叠，不应称为完全独立的留出验证。

## 六、还缺什么，以及不需要做什么

### 必须明确的证据缺口

1. **独立正常对照的联合通过率**：使用未参与阈值计算的新随机种子及新目录，保持相同条件、reads和采样配置，先冻结候选规则再检查；保留全部通过/失败批次。
2. **分布异常识别检查**：对明确标注的本地故障样本（例如塌缩为同一个状态、错误位序等）检查规则响应。格式错误能拦截，不等于格式合法但分布错误也能拦截。此类人工故障样本只能用作规则/软件测试，不能作为能源预测或真机实验结果。
3. **参考是否足够可靠的说明**：有限sweeps的SA是操作性参考，不能仅因beta估计接近1便宣称精确抽样；需说明其局限和适用范围。

补验证前，应明确批次数与停止点，不能“不断增加直到通过”。如果改变阈值，应记录理由、保留旧规则，并用新的留出数据再验证。没有必要承诺95%的联合通过率；若只作为探索性的额度停止规则，应明确接受它未被充分标定的风险，不能换成更强统计主张。

### 本报告不要求

- 不要求重训FA-BM-VAE、补10个训练种子、引入新数据集或增加SDWPF。
- 不要求提交真机来完成本地对照检查。
- 不要求先取得论文性能提升再批准工程连通测试。
- 不把本次未运行的独立采样或异常对照标成“已完成”。

## 七、审批与运行顺序

1. 本报告当前为审阅稿：保留 `thresholds_approved=false`。
2. 负责人明确选择接受或暂缓当前探索性工程规则并承担已说明的限制；此前“独立验证缺失”的表述已由后续严格 v1-v4 历史审计替代，不能通过签字把有限证据变成硬件保证。
3. 形成最终人工确认记录后，保留当前协议和v7，另建批准后的协议副本及完整准备包，核对所有身份、路径、参考文件与矩阵。
4. 平台权限/路由/计费事项确认后，另行批准1个条件、100 reads的canary；成功后停下审阅实际返回。
5. 只有再次明确批准继续、规则有效且输入身份一致，才执行剩余VS；完整真实VS通过后选定gain，才可考虑TEST。

当前程序允许独立批准未完成质量审批的canary，但它是隔离工程探针。若后来修改协议，不能保证其记录直接计入新协议。为节省额度，当前推荐在最终协议冻结之后再做首次采样，不关闭哈希检查、不改写旧响应身份。

原始失败或状态未知任务必须先对账，不能删除pending intent后重投，也不能为了跑通而关闭严格响应验收。

## 八、人工审阅意见（本报告未代填）

审阅人/责任人：________________

审阅日期与时区：________________

决定（由人明确填写）：继续补本地验证 / 接受有明确局限的探索性工程规则 / 暂缓 / 其他：________________

接受的指标与阈值版本、依据：________________

尚未完成的验证和接受的风险：________________

是否另行批准首次100-read提交：________________（本报告默认不批准）

可用确认文字示例，仅供人工填写，不能由程序自动采纳：

> 我已阅读指定版本报告及其证据缺口，确认采用所列指标、聚合单位、阈值与停止条件，并记录采用该规则的理由和局限。此确认不代表硬件性能已验证，也不批准自动扩展任务或重试未知任务。首次真机提交须另行明确授权。

## 九、解释风险筛查（11/11项）

| 风险类型 | 本次判断 |
|---|---|
| 辛普森悖论/分组掩盖 | 未建立相关性反转结论；已提示按gain顺序累计的组构成影响，不能以均值推断各条件 |
| 生态谬误 | 10条件均值合格不代表每条件合格；需保留条件级诊断 |
| Berkson选择偏差 | 未进行因果相关分析；固定10条件不能代表所有天气/时段 |
| 碰撞变量偏差 | 当前不是因果调整模型，不适用；不据此宣称排除所有选择偏差 |
| 基础率忽视 | 正常对照通过率不等于真实硬件可靠概率；异常基率未知 |
| 回归均值 | 不从“选择失败批次后重跑变好”推断改进；当前无此验证证据 |
| 幸存者偏差 | 必须保留所有失败/无效任务；本轮未按通过状态删参考批次 |
| 多重搜索 | 四指标联合规则和三个gain不能被描述为单项95%保证；不得只报告通过项 |
| 分析路径自由度 | 候选阈值、样本面板和审批须在正式结果前固定；修改需留痕 |
| 相关即因果 | 本地对照接近不能说明量子硬件改善预测或具有量子优势 |
| 反向因果 | 当前不适用因果方向检验；不得用TEST结果倒推gain或阈值 |

总体置信判断：**CAUTION**。主要限制是独立质量验证缺失和联合规则尚未标定，而不是已发现经典训练结果为假。

## 附录A：测试与复算边界

本次实际运行：

```powershell
D:\anaconda\envs\wsy\python.exe -m pytest tests/test_kaiwu_13k_protocol.py tests/test_kaiwu_batch_driver.py tests/test_opsd_quality_gate.py tests/test_submit_kaiwu_safety.py -q
```

结果：62 passed，14 warnings，22.16秒。警告为依赖弃用提示；测试未包含付费集成测试。测试中的模拟响应不是硬件证据。

数值复算步骤：按gain与repeat_pair分组，对每组10行已有条件统计等权平均；对15个均值用NumPy默认线性插值计算q95；逐一删除一批后重算q95；用未删除的全样本q95统计15批四指标同时达标数。未将这项复算称为新的独立实验。

整体状态保持ANALYZED：已复算汇总和分位数、已重跑工程测试，但没有从头重跑SA或完成独立留出验证。

## 附录B：关键快照哈希

| 文件 | SHA-256 |
|---|---|
| 当前协议JSON | `247ae50634bdf0eee8684b58a671b6901615899a82dd0e2325cda3b9d9f72612` |
| v7 preparation_manifest.json | `ab8b1370d023841b455ca0b4b29aa068107feefc94edad961ca57009ef31cca5` |
| v7选定 quality_gate_reference_audit.json | `1af1db48b79757eeb61e2fba373acee84f8aae0ccddcd0b3d38487f48d63db6d` |
| opsd_quality_gate.py | `ff7a00e410e8389c75b0702d2658ddac350a624da49daa4ef40d9ea287e80f40` |
| kaiwu_13k_protocol.py | `e21e807961a85c6cb7d97cb437da44d783abd163bd8a3bd70c5dbb377751adf7` |
| run_kaiwu_13k_sampling.py | `3b4325fd44f3c423c8978c89bb7d688f7022f470fb59c709d18a27cecd8a0ee1` |

## Independent zero-credit validation update (2026-10-18)

This update is an independent local validation of the frozen OPSD quality rule. It did not retrain the model, read new TEST metrics, call the Kaiwu/physical platform, approve thresholds, or change the protocol. The complete machine-readable output is in `export/opsd_sa_quality_validation_20261018_v2`; the first failed aggregation run is preserved in `..._v1` and was not reused.

### Engineering checks passed

- Ten new disjoint batch seeds were fixed in advance: `303000` through `303009`. The derived condition seeds use `batch_seed + gain*1000 + condition_index` and are above the checked historical seed range.
- The validation used the same frozen ten-condition VS panel, gains `125/150/200`, 100 reads, 100 sweeps, and the `repeat_00` frozen reference samples from `quality_reference_manifest.json`.
- Source-problem hashes and frozen-reference hashes were checked before sampling. The new output contains 300 normal-control files and 600 explicitly labeled anomaly-control files.
- The protocol remains `thresholds_approved=false`, `test_data_used=false`, `hardware_claim=false`, and `physical_platform_used=false`.

### Quality validation results

The formal validation unit is one gain and one batch, aggregated over the same ten VS conditions; it is not an independent count of individual reads.

| Control | Gain-batches | Joint pass | Joint pass rate |
|---|---:|---:|---:|
| Normal frozen-reference control | 30 | 24 | 80.00% |
| Collapse anomaly control | 30 | 0 | 0.00% |
| Reverse-bit-order anomaly control | 30 | 0 | 0.00% |

For the normal control, failure counts were Energy Wasserstein `0`, effective-beta error `0`, edge-moment MAE `5`, and magnetization MAE `1`. The collapse and reverse-bit-order controls failed all four metrics in all 30 gain-batches. These anomaly files are software diagnostics only and cannot be used as physical or paper results.

Condition-level diagnostics are preserved in `condition_diagnostics.json`; they are not a second formal gate. The mean condition-level joint-pass rates across batches were 0.35 for gain 125, 0.25 for gain 150, and 0.18 for gain 200.

### Decision and remaining risks

**DEFER / PENDING MANUAL DECISION.** The normal control is compatible with the existing exploratory candidate thresholds at an 80% gain-batch joint-pass rate, and the two severe anomaly controls are detected. This does not establish a 95% guarantee, hardware reliability, or a physical-platform success rate. The four thresholds remain unchanged and unapproved. A reviewer must still decide the acceptable false-rejection risk, aggregation unit, and coverage of subtler hardware faults. A canary format/identity approval remains separate from formal quality-threshold approval and cannot unlock batch VS or TEST.

## Seed-allocation correction and independent validation v4 (2026-10-18)

The v2 result is retained but is **not** accepted as independent-random-stream evidence: its legacy formula produced 300 files from only 57 unique recorded seeds because batch and condition indices could collide. An earlier note that v3 lacked a complete prior-directory audit is superseded by the strict v1-v4 audit below; v3 remains the corrected independent run and v4 remains its reproduction/audit completion.

The corrected v3 run uses 300 actual seeds, `303000` through `303299`, allocated by `seed_base + ((gain_index*10 + batch_index)*10 + condition_index)`. The complete mapping was frozen in `export/opsd_sa_quality_validation_20261018_v3/seed_plan.json` before sampling. Historical reference, v1, and v2 recorded seed sets were checked dynamically; counts were 300, 57, and 57 unique seeds, with no intersection with the new plan. v4 was then audited against v3: all normal sample arrays, sample files, seed mappings, protocol/preparation/threshold hashes, gains, reads, sweeps, and thresholds matched exactly. Therefore v4 is a reproduction and audit completion, not additional independent evidence.

| Control | Gain-batches | Joint pass | Joint pass rate |
|---|---:|---:|---:|
| Corrected normal frozen-reference control | 30 | 23 | 76.67% |
| Collapse anomaly control | 30 | 0 | 0.00% |
| Reverse-bit-order anomaly control | 30 | 0 | 0.00% |

Corrected normal-control failures were Energy Wasserstein `0`, effective-beta error `1`, edge-moment MAE `5`, and magnetization MAE `3`. The corrected results do not adjust thresholds or imply that the rule has been approved. The v3/v4 evidence remains local SA evidence only, with `hardware_claim=false`, `physical_platform_used=false`, and `eligible_for_paper=false`; the two runs must not be pooled as 600 independent samples.

**Updated decision: DEFER / PENDING MANUAL DECISION.** The seed design and historical non-overlap checks now pass, but the 23/30 normal joint pass rate and limited anomaly coverage remain human review risks. No real task or TEST unlock is authorized.

## Final pre-canary audit (2026-10-18)

The read-only audit output is `export/opsd_sa_quality_validation_audit_20261018_v3/historical_seed_audit.json` and `historical_seed_audit.md`. It requires and audits the threshold reference plus v1, v2, v3, and v4; missing, damaged, or seed-less normal files fail closed. Deterministic anomaly files are excluded from random-sample counts. The same strict audit is now executed by `run_independent_validation()` before any new sampling call.

The audit confirms that v3 and v4 have identical normal sample arrays, normal sample files, seed mappings, protocol hash, preparation hash, threshold source, gains, reads, sweeps, and threshold values. v4 is therefore a reproduction/audit completion of v3 and adds zero independent evidence. The 23/30 normal result is reported once, not pooled across v3 and v4. v1/v2 remain historical legacy runs with seed reuse and are not independent evidence.

The local result remains 23/30 normal controls passing (76.67%); both deterministic severe anomaly controls were rejected. This is not a hardware success rate and is not a 95% quality guarantee. The audit keeps `thresholds_approved=false`, `hardware_claim=false`, and `physical_platform_used=false`.

Current frozen-input checks found no change in the model/data/v7 preparation artifacts during this audit: protocol SHA-256 `247ae50634bdf0eee8684b58a671b6901615899a82dd0e2325cda3b9d9f72612`, v7 preparation manifest SHA-256 `ab8b1370d023841b455ca0b4b29aa068107feefc94edad961ca57009ef31cca5`, model checkpoint SHA-256 `0281db5b136c21e5c3327b9a30c4f31dadeb4c04b90998a28670fc57ff347a5c`, dataset SHA-256 `507ed5ae709c05f01e6e67209e5ae08b2567f96755c84ed2444dc5cd7d296555`, and current v7 matrix digest `f724e3be3116e9e675e42784c3a9c5caf19680132d85f6c20c829c5a31683210` over 30 matrix files. Historical selective-retention decisions are not fully recoverable from stored artifacts and remain marked unknown.

Local safety checks remain separate from hardware evidence: invalid responses stop the batch without retry, unknown task state blocks resubmission, canary approval is separate from batch approval, and no real Kaiwu task was submitted. `thresholds_approved=false` remains unchanged.

**First 100-read canary status: BLOCKED pending manual approval.** Before execution, a reviewer must confirm the account/project permission, SPQC-550 route, `SAMPLING` mode, SDK/API compatibility, credits/read conversion and failure billing. After one approved canary returns, execution must stop for identity, hash, returned-read, spin, auxiliary-spin, and decoded-scenario review; canary success cannot unlock batch VS or TEST.

快照用于区分受审版本，不意味着修改文件后应手工替换旧哈希使其“通过”。原始实验文件通常位于Git忽略的export目录；报告上传GitHub本身不能替代本地原始证据归档。

## 2026-10-20 operational addendum

This historical report is retained as written. Current status is recorded in [OPSD current operations](OPSD_CURRENT_OPERATIONS_20261020.md): v8 uses int8-matrix SA for the latent diagnostic reference; floating-point logical SA is a separate backend. The four historical latent thresholds remain diagnostic reference values and are not validated acceptance limits. A complete valid real VS panel selects gain by mean Raw VS CRPS, with exact ties choosing the smaller gain; TEST needs independent explicit confirmation. No real task or TEST metric has been run as part of this addendum.
