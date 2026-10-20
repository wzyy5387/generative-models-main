# 冻结模型推理后端对照：待确认方案与复用审计

> 后续修订：质量规则代码已按 [2026-10-19 修订说明](OPSD_QUALITY_POLICY_REVISION_20261019.md)
> 改为工程硬门槛、潜变量诊断和 Raw VS CRPS 选择；本文其余内容保留原方案审计时点。
> 这不表示三后端实验已经运行，也不表示新协议/真机提交已获批准。

**状态：仅方案与证据盘点；实验结果均为 `NOT RUN`。**

本文件只审计本地已有文件并拟定后续方案。本轮未训练、未新增 SA 采样、未计算 TEST 指标、未调用真机，未改阈值及其批准状态，也未启动或批准后续采样。既有经典多模型、多种子主结果保留原样，不以单检查点结果覆盖或替代。

## 1. 冻结对象与身份

建议固定 OPSD `FA-BM-VAE` 负相 SA seed 0 检查点：
`opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0`。这是**已经训练好的模型**；对照只替换训练后条件 Ising 潜变量的推理采样后端，不比较训练后端，也不构成量子优势检验。

| 对象 | 文件 | SHA-256 |
|---|---|---|
| 检查点 | `export/qbm_vae_opsd-wind/opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0.pickle` | `0281db5b136c21e5c3327b9a30c4f31dadeb4c04b90998a28670fc57ff347a5c` |
| 检查点配置快照 | `export/qbm_vae_opsd-wind/opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0.json` | `c1d84154cb1232258bee93b2200ee9f4ec1f9017dbdc6165af1300cf2520ff1b` |
| 训练方案配置 | `configs/paper/fa_bm_vae_opsd_wind_sa_negative_v1.json` | `e3aa906a675b6d1fa359c55b9b489a668a37b9e7529ce97b40ef972d69729a8a` |
| 数据包 | `GEFcom2014/data/external/opsd_wind_daily.npz` | `507ed5ae709c05f01e6e67209e5ae08b2567f96755c84ed2444dc5cd7d296555` |
| 当前硬件准备协议 | `configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json` | `247ae50634bdf0eee8684b58a671b6901615899a82dd0e2325cda3b9d9f72612` |

保存的配置确认负相后端为 SA；训练方案为 80 epochs、48 个潜变量、100 场景，checkpoint 选择规则记为最低 VS 重构损失，未用 TEST 选择。数据按日期切分：LS 2016-01-01 至 2018-11-01，VS 2018-11-02 至 2019-01-30（90 天），TEST 2019-01-31 至 2019-04-30（90 天）。缩放器仅用 LS 拟合；v7 `frozen_model_audit.json` 记录 scaler 参数哈希：x_mean `306d84975b76684f9e5c29e56bb25c2edc38149a3f25ff9efc4dab1dbbe0bd5a`，x_scale `ba814967f3ad27866064cf182d0b06acd7861414e71e73c9c6b108e7bd4cc1bb`，y_mean `9a27bf565acd33c9f8b899dd6f710568778bb58e41ba6da2557e2281f0a93050`，y_scale `b7aaed62e26b8fa132ce61a63cc856983b71475b238f44e02fd0ba4ace6c4b97`。Scaler 实际用于历史逐条预测的证明仍须与对应生成记录绑定。

v7 选择清单固定 10 个 VS 日期，索引 `[65, 7, 40, 33, 14, 26, 47, 17, 35, 74]`，选择种子 2026。每个条件目标 100 条场景。量化候选 gain 为 125、150、200；矩阵为 48 个逻辑自旋加 1 个辅助自旋，`numpy.rint` 半偶数舍入，无 clipping。准备目录保留了每个实例矩阵 SHA-256，整个 30 矩阵集合摘要为 `f724e3be3116e9e675e42784c3a9c5caf19680132d85f6c20c829c5a31683210`（v7 审计记录）。

关键原始文件哈希已核对：v7 preparation manifest `ab8b1370d023841b455ca0b4b29aa068107feefc94edad961ca57009ef31cca5`；VS 选择清单 `47e3c2e82352a3f8992830c4e8c4f47a08f27164ea1e04cfafd6046e34290180`；VS 索引文件 `c9f303ecc181594cabbd51a6d7886646430c5dcccec67cd89f4cace02c3b7ac2`；gain-125 的 `vs_000.npz` `9c7e653a4c0e189b21920ab123be6c230774294f1249db56044531bb18d49a4a`；对应 JSON `665af9bae2ce89cd2dee9b2419ccce28b53582eb6881ec4bffe5eaf63f5c1b8a`。seed-0 历史 VS pickle 的 SHA-256 为 `b4a58548aa91976553d2735c8b358b6ef23965485268a88dbcc97f73dd2ef48f`。逐实例源矩阵与 SA 参考样本哈希见 `quality_reference_manifest.json`（该 manifest SHA-256 `788a63d850483d7ec5f16b83a63a9dae6cbc116fb39568535e103d2f480d7611`）。

## 2. 最小三路径设计（待确认）

| 路径 | 固定检查点输入 | 推理采样器 | 每个冻结日期场景数 | 状态 |
|---|---|---|---:|---|
| A | 原始浮点逻辑 Ising 参数 | 浮点逻辑 SA | 100 | `NOT RUN` |
| B | 与 C 完全相同的硬件整数矩阵及 gain | 经典 SA | 100 | `NOT RUN` |
| C | 同一硬件整数矩阵、同一 gain | SPQC-550 真机 | 100 | `NOT RUN` |

首选沿用已有 v7 VS 的 10 个冻结日期作为 VS 条件面板；是否确认这 10 天为正式协议由人工决定。A 使用逻辑矩阵；B、C 对每个日期使用相同量化矩阵、gain、变量映射和辅助自旋解码。若按既有候选网格比较 gain，则 B、C 都逐个覆盖 125/150/200，不能让 B 的矩阵与 C 的矩阵不一致。Gibbs 仅作为可选扩展，当前不纳入工作量或自动新增实验。

三路径使用同一 decoder、同一 LS scaler、同一日期条件、同一后处理和观测噪声开关/随机种子映射；建议固定 `observation_noise=True` 并对同一日期采用共同噪声种子，以减少解码随机性造成的差异。v7 本地链记录了 100 reads、SA 100 sweeps，噪声开启且 seed base 为 2026；但该设置本身不证明路径间配对已严格执行，复用前仍须核查原始数组及解码实现。

对照只估计**该固定检查点在推理阶段替换采样后端后的差异**。它不估计不同训练方法的优劣，不代表整个模型族，不证明硬件普遍性能，更不检验量子优势。

## 3. 已有 VS 与推理产物复用审计

| 产物 | 判定 | 已核对内容与限制 |
|---|---|---|
| v7 `export_vs_gain_{125,150,200}` 的 30 个条件 JSON/NPZ、VS manifest 和索引 | **可直接复用**（仅作固定输入） | 10 个 VS 条件 × 3 gains；检查点 ID、条件日期、逻辑/硬件维度、gain、矩阵哈希和量化参数可追溯。它们是问题输入，不是采样响应或性能结果。 |
| v7 `quality_gate_reference_v7_revalidated/samples` 中 10 repeats 的本地经典 SA 原始响应 | **可直接复用**（路径 B 的本地 SA 证据候选） | manifest 记录每实例 100 reads、10 repeats，来源为冻结矩阵，采用本地 SA；不是物理设备。正式配对前需按实际 manifest 将重复、日期、gain和样本映射核对齐。 |
| v7 `quality_gate_reference_v7_revalidated` 的浮点逻辑 SA 参考样本及其解码 | **需要重新解码**（路径 A 配对版本） | 文件可用于解释既有零信用质量门；需以三路径统一 decoder、同一噪声方案重新导出逐日期预测。只重解码存档原始自旋，不追加 SA reads。 |
| v7 `preparation_manifest.json` 中的 `local_validation` 对比行 | **证据不足**（不得作为正式路径 A/B/C 结果） | 文件明确标注 `backend=SA/mock response`、`hardware_claim=false`、`physical_platform_used=false`；有 10 日期、100 场景形状和 Raw 指标，但 quantized 行来自 mock response。只能作为离线链路/格式检查。 |
| v1/v2 `mock_responses`、`imported_mock` 及 v7 `quality_gate_reference_v7_revalidated_refresh*` 离线 mock 链 | **证据不足**（不可称真机） | 能用于校验身份绑定、解码、质量门和 dry-run；不得作为路径 C、真实 VS gain 选择或 TEST 准入证据。v7 未发现 `real_responses` 目录；历史清单明确 `physical_platform_used=false`。 |
| 旧主结果 `export/qbm_vae_opsd-wind/scenarios_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_100_VS.pickle` | **需要重新解码** | 文件名与同目录评估代码确认 100 场景/日期；所属 VS 切分为 90 天（2018-11-02 至 2019-01-30）。文件哈希已记录，但模型条件/逐日期原始 latent spins 未由该 pickle 本身证明。历史生成 SA reads/sweeps、解码噪声开关及种子、逐日期 scaler 和后处理记录不足以验证本次统一方案；训练负相配置中的 1 read/20 sweeps 不能当作推理参数。若归档无原始 spins，则此行转为“需要补跑推理”。 |
| 旧主表中的其他模型/种子结果及三种子平均 | **可直接复用**（仅保留原主结果） | 保持为已有经典多模型、多种子比较，不覆盖；不能把跨种子平均作为 seed-0 单检查点的真机配对对照，也不能从中推断本次三路径差值。 |
| 任何路径 C 的真实 SPQC-550 VS 原始响应与逐日期解码 | **需要补跑推理** | 当前没有物理平台返回证据。需未来另行批准后按选定协议采集；本方案不发起任务。 |

**解码核对结论：**数据范围、LS scaler 来源、10 个 VS 日期、场景目标数、量化矩阵 identity 与 gain 网格可由 v7 清单确认。历史保存预测缺少足以证明完全配对的逐条件自旋/噪声/后处理链；v7 离线验证的 100 reads、SA 100 sweeps、噪声为开、seed base 2026 和 Raw 指标均有记录，但它将本地 SA 与 mock response 放在同一验证流程中，绝不等于真机。v7 的 decoder 和统一 scaler 可用作将来统一重解码实现，但本轮未执行重解码或重算指标。

## 4. 统一评估协议草案

- 冻结同一批 VS 日期、同一条件输入和每条件 100 场景；TEST 若之后获准，使用冻结 TEST 日期且每条件 100 场景。
- 对 A/B/C 固定 decoder、LS scaler、观测噪声开关与种子映射、辅助自旋处理和后处理；逐日期保留未经校准的 **Raw** 为主分析。校准/ECC 等仅可作为明确标识的次级分析，不能替代 Raw。
- **CRPS 为主要指标**；Energy、Variogram、Ramp CRPS、覆盖率及偏差为辅助指标。输出一行一日期/路径/gain/指标的逐日期文件，再按预测日期汇总路径内结果及同日期差值。统计单位是预测日期，不是 reads、场景 draw 或同一天的自旋重复。
- 若采样随机性需要重复，repeat 只用于估计 sampler 稳定性；不得把 100 reads 当 100 个独立日期，也不得把同日重复伪装为样本量。
- TEST 量化 SA 必须使用真实 VS 最终选出的 gain 对应的同一量化方案。mock 得到的 gain、offline mock freeze 或 dry-run 不能替代真实 VS 选择结果。

### 三路径结果空表（所有值尚未运行）

| 日期 | Gain | A 浮点 SA CRPS | B 量化矩阵经典 SA CRPS | C 同矩阵 SPQC-550 CRPS | Energy | Variogram | Ramp CRPS | 覆盖率 | 偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 冻结 VS 日期 1–10 | 待确认 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |

## 5. 需要人工确认的协议决定

1. **四项潜变量质量指标的用途：**Energy Wasserstein、effective-beta error、edge-moment MAE、magnetization MAE，哪些是工程有效性检查，哪些是质量门，哪些只作科学诊断？当前阈值均未获批准。
2. **gain 选择方法：**是否沿用 VS 的 125/150/200 候选网格，如何依据真实 SPQC VS 结果选择最终 gain，以及并列/失败时如何处理？本地 mock 选择不得代替真实 VS 结果。
3. **分布指标超出阈值：**确定是停止、复核、重复验证还是保留并作为科学偏差报告；不可在查看结果后临时放宽阈值。
4. **TEST 准入：**须明确哪些 VS 证据、审批人/审批记录和工程检查全部通过后才可提议 TEST。当前 `thresholds_approved=false` 保持不变；本文件不批准继续采样，也不自动解除 TEST 锁定。

判定层次应分开：检查点/矩阵/任务身份或哈希不符，日期/维度/read 数不符，解码失败，缺 scaler 或结果文件损坏，均属**工程错误硬停止**，不得继续该批次或自动重试。分布指标偏离则属**科学诊断/人工审议**；不应伪装成工程错误，也不能由脚本自动改变阈值批准状态。

## 6. 缺失工作与本地计算量

| 待办（须先确认方案；本轮均 `NOT RUN`） | 预计规模 |
|---|---:|
| 核对并统一重解码可复用的 A/B 原始自旋样本 | 10 日期 × 100 场景/路径；本地 CPU/GPU 时间未知 |
| 路径 B 对候选 gain 做经典 SA（若现有样本映射不满足配对） | 最多 3 gains × 10 日期 × 100 reads = 3,000 reads；是否需补采样取决于原始样本审计，当前不运行 |
| 路径 C 真实 VS（若要在三个候选 gain 中选 gain） | 3,000 真机 reads；费用/耗时/配额换算未知，需单独授权 |
| 选定 gain 后的 TEST 路径 B | 90 日期 × 100 reads = 9,000 本地经典 SA reads；尚未计算指标 |
| 选定 gain 后的 TEST 路径 C | 90 日期 × 100 reads = 9,000 真机 reads；当前 TEST 锁定，未批准、未运行 |
| 逐日期统一解码与 CRPS/辅助指标 | 每路径每条件 100 场景；本地运行耗时未知 |

现有 v7 工程准备曾报告 VS 3,000 + TEST 9,000 的计划预算及 credit 换算未核实；那是旧硬件准备计划，不是本方案已批准预算。不得把 mock、离线 SA 或旧的 dry-run 计入真机 reads。方案获确认之前不做上述实验或计算，完成此文档后等待确认。

## 2026-10-20 operational addendum

This historical report is retained as written. Current status is recorded in [OPSD current operations](OPSD_CURRENT_OPERATIONS_20261020.md): v8 uses int8-matrix SA for the latent diagnostic reference; floating-point logical SA is a separate backend. The four historical latent thresholds remain diagnostic reference values and are not validated acceptance limits. A complete valid real VS panel selects gain by mean Raw VS CRPS, with exact ties choosing the smaller gain; TEST needs independent explicit confirmation. No real task or TEST metric has been run as part of this addendum.

## 2026-10-20 provenance addendum for historical SA-negative seeds 0--2

This addendum is an evidence audit only. It does not rewrite the historical JSON
files, retrain a model, regenerate scenarios, or alter the frozen hardware package.

Directly verified in the three frozen checkpoints/configuration snapshots:

- all three checkpoint, loss, VS-scenario, and TEST-scenario files exist;
- `negative_phase_backend=sa`, `negative_steps=5`, one negative read,
  20 sweeps, 20 burn-in sweeps, and `sampler=sa` are recorded;
- `decoder_anchor=true`, `anchor_source=context`, `in_size=24`, and
  `cond_in=57` are recorded;
- checkpoint runtime metadata records an SA backend with 20 sweeps;
- the dataset bundle is
  `GEFcom2014/data/external/opsd_wind_daily.npz`.

Supporting, but not complete historical execution evidence:

- the retained protocol records 80 epochs and minimum VS reconstruction loss as
  the checkpoint rule;
- the current source path selects `VS reconstruction` and the historical loss
  arrays contain 80 rows with no finite TEST-loss values;
- the three checkpoint masks independently contain 48 nodes and 172 upper-
  triangular nonzero undirected edges.

The per-seed JSON snapshots all retain `graph_edges=99`. This conflicts with the
checkpoint masks; the cause of the stale field was not recorded. The historical
JSON files remain unchanged and the checkpoint mask is authoritative for any
future reconstruction.

Not recoverable from the repository and searchable execution records:

- the exact original command for each of seeds 0, 1, and 2;
- contemporaneous stdout/stderr and environment/source commit;
- an independent per-run best epoch or checkpoint-selection event;
- a direct historical record of whether TEST was inspected manually after
  training. The available evidence supports the training-time VS-only path,
  but does not prove that TEST was never viewed for any later decision.

The README command is retained only as a reproduction template, not as direct
historical command evidence. The working tree was dirty at audit time; the
current commit therefore does not by itself reproduce the historical source.

The read-only recomputation of the three seed-level checkpoint, configuration,
loss, VS-scenario, TEST-scenario, and graph-mask hashes is stored in
`export/opsd_sa_negative_provenance_audit_20261020/checkpoint_hash_audit.json`
(audit SHA-256
`dd3166b4be6b5c894ae8c298ba3ae50fccab47328ffc97d00dd87490803b060b`). It
observed 48 nodes and 172 upper-triangular nonzero undirected edges in every
checkpoint, while every historical JSON still records `graph_edges=99`; the
cause remains unknown and the historical JSON files were not edited. The loss
arrays have 80 rows and zero finite TEST-loss entries for all three seeds. This
recomputation created no new TEST metrics and did not invoke a platform.
