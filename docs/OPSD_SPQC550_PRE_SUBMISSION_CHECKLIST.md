# 旧 OPSD / SPQC-550：首次采样前收尾

> 2026-10-19 修订：以下 v7/硬阈值描述属于历史状态。当前按
> [新质量规则](OPSD_QUALITY_POLICY_REVISION_20261019.md) 执行：工程有效性硬停止、四项潜变量偏差仅诊断、完整真实 VS 按 Raw CRPS 选 gain。
> 新协议未批准，且须生成新准备包，不能拿 v7 直接提交。

更新日期：2026-10-18。本文是本地工程验收与操作要求，不是人工采样授权，也不是真机实验证据。

## 当前范围

- 模型、旧 OPSD 数据、历史经典结果不变，不重训。
- v7 是准备材料版本，不是模型版本。当前协议仍未批准质量阈值。
- VS：10 条件 × 3 gains × 100 reads = 3,000 reads；TEST：90 日期 × 100 reads = 9,000 reads。
- 首次 100-read canary 应在最终冻结协议下运行并计入 VS 预算。15,000 Sampling Credits 不等于已经核实的 15,000 条有效样本；扣费换算待平台确认。
- 不自动批准阈值、不提交真机、不重试未知任务、不动用备用额度。

## 已修复的提交保护

1. 当前响应严格验收失败，立即终止 VS/TEST 批次，保留原始响应、失败记录和 pending intent，不提交下一任务。
2. `--canary` 必须指定冻结 VS 实例、gain=125、100 reads、`--limit 1`；真实运行另需 `--confirm-canary`。
3. 非 canary 真实提交必须另带 `--confirm-batch`，并通过质量规则校验。漏写 `--canary` 不会仅凭通用提交确认进入批量运行。
4. canary 与批量确认不能混用；孤立的 `--confirm-canary` 被拒绝。
5. 正式规则必须已批准、有版本号、四个误差阈值完整且有限、非负。测试专用规则不能授权正式采样或正式 gain 冻结。
6. TEST 仍须由完整真实 VS 响应选定 gain；canary 成功与本地 mock 均不能替代此条件。

这些检查只保证相应的软件控制流程，不验证人工批准的真实性、账号权限或硬件采样质量。

## 协议哈希问题：采用“先冻结，后采样”

程序核验整个协议 JSON 的 SHA-256；修改阈值、批准状态，甚至文件字节变化，都会改变协议身份。不要关闭此检查、改写旧响应身份或悄悄刷新旧 manifest 哈希。

最小解决方案不增加独立审批文件系统，而是调整正式流程：

1. **先审阅离线质量参考**：15 个 gain-level 参考批次的 q95 只是候选工程界限，不等同于经过验证的 95% 保证。查看阈值稳定性、参考 SA 配置及适用范围；若证据不足，保持未批准，必要时只补本地参考。
2. **由研究负责人明确批准规则**：记录批准人、日期、依据、四个阈值和停止规则。不得由程序把 `false` 自动改为 `true`，不得根据之后的真机结果放宽阈值。
3. **保留当前协议及 v7**：另建经批准的协议副本与新的准备目录。此时尚无正式采样，重新准备材料不消耗真机额度。不要覆盖 v7，也不要盲目复制后仅修改哈希。
4. **完整验收新包**：核验 checkpoint/数据/逻辑参数/量化矩阵/参考样本及哈希、任务名、10 条件三 gain 网格、90 个 TEST 逻辑条件；必须确认模型和矩阵没有因审批意外改变。更新全部引用路径和质量参考清单，再验证真实入口的 mock 链路及 canary dry-run。
5. **固定路径后再提交 canary**：协议、准备包、响应目录、checkpoint 目录此后保持不变。后续批量恢复使用相同目录和 `--resume`；不要回到 v7 或离线 mock 目录。

技术上仍支持在质量阈值未批准时单独批准连通测试，但这只是一项隔离的工程探针。若之后修改协议，不能承诺其记录能直接计入新协议；因此，在当前额度约束下不推荐先跑再改协议。若必须这样做，应先另行实现并测试显式身份迁移/兼容性核验，不能伪装为普通续跑。

## 当前 v7 的零额度检查命令

工作目录必须为 `D:\研究生作业\组会4\generative-models-main`。以下仅 dry-run，不提交任务，也不代表批准 v7 正式采样：

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.run_kaiwu_13k_sampling `
  --stage vs `
  --protocol configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json `
  --preparation export/opsd_sa_negative_kaiwu_preparation_v7 `
  --output-dir export/opsd_sa_negative_kaiwu_preparation_v7/real_responses `
  --checkpoint-dir export/opsd_sa_negative_kaiwu_preparation_v7/checkpoints `
  --canary --canary-instance-id opsd-wind_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_vs_000 `
  --limit 1 --resume --dry-run
```

预期：`tasks=1 reads=100 dry_run=True`，硬件声明为 false。正式命令需在最终批准的新包验收后重新给出，不应直接把这条命令改成 execute。

## 必须由人确认的外部事项

- 当前 SDK/许可证和账号有 SPQC-550 的 SAMPLING 权限；机器简介和本地 profile 不是路由证明。
- 每次请求的额度换算、失败或返回不足时如何计费；保存首次任务前后余额与平台任务记录。
- 首次返回核对任务编号、100×49 自旋、矩阵哈希、元数据，再验证辅助自旋处理和冻结解码器生成场景。
- 超时、未知状态或无效响应：立即停下，先在平台核实原任务，不自动重投。
- canary 成功后仍须人工确认是否继续 VS；质量规则批准不等于硬件采样一定合格。所有 gain 不合格就停，不花 TEST 额度“碰碰运气”。

## 本地回归命令（不含付费集成测试）

本次实际验收：62 项测试通过；另核验 v7 协议身份、30 个 VS 输入矩阵、逻辑参数与参考样本哈希，以及 checkpoint/config/dataset 文件哈希，均通过。既有离线 gain freeze 重新校验通过，仍为非硬件结果。这不是全仓库审计，也不保证平台返回或科学结论。

```powershell
D:\anaconda\envs\wsy\python.exe -m pytest tests/test_kaiwu_13k_protocol.py tests/test_kaiwu_batch_driver.py tests/test_opsd_quality_gate.py tests/test_submit_kaiwu_safety.py -q
```

不要为本地回归启用 `FA_BM_VAE_RUN_REAL_KAIWU_TESTS`，也不要运行付费集成测试来代替上述验收。

## 2026-10-20 operational addendum

This historical report is retained as written. Current status is recorded in [OPSD current operations](OPSD_CURRENT_OPERATIONS_20261020.md): v8 uses int8-matrix SA for the latent diagnostic reference; floating-point logical SA is a separate backend. The four historical latent thresholds remain diagnostic reference values and are not validated acceptance limits. A complete valid real VS panel selects gain by mean Raw VS CRPS, with exact ties choosing the smaller gain; TEST needs independent explicit confirmation. No real task or TEST metric has been run as part of this addendum.
