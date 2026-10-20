# OPSD 真机质量规则修订（2026-10-19）

## 范围与状态

按用户要求修改质量规则及离线校验代码，不训练、不提交平台任务、不计算新的 TEST 指标、不消耗采样额度。
历史模型、数据和 v7 目录不修改。单元测试中的合成数据与模拟物理元数据仅用于工程验证，不是论文结果或真机证据。

协议版本：`fa-bm-vae-opsd-spqc-550-sa-negative-v3-vs-crps`。
质量规则版本：`opsd-sa-vs-quality-v2-diagnostic-raw-crps`。
当前 `policy_approved=false`，不是采样授权。历史协议保留兼容处理，不把旧阈值门的 mock 验收当成新规则验收。

## 冻结规则

| 层次 | 新规则 |
| --- | --- |
| 工程有效性 | 任务、矩阵、模型/准备包/协议身份及哈希、49 自旋格式、100 reads、响应文件完整性必须通过。无效响应保存证据，立即停批，不自动重试。 |
| 解码有效性 | VS 自旋经辅助自旋还原后，使用冻结解码器、LS 拟合缩放器、冻结 VS 条件和标签重算预测；形状或有限性不合格停批。保存失败记录并阻止自动续跑。 |
| 潜变量诊断 | energy Wasserstein、effective-beta error、edge-moment MAE、magnetization MAE 全部重算保留；旧参考阈值与超限状态只作诊断，不淘汰候选 gain。unique-state fraction 也只作诊断。非有限指标不是正常超限，仍阻止完成验收。 |
| gain 选择 | 同一 10 个 VS 日期，125/150/200 三个 gain 各 100 reads；完整 30 个响应均有效后，选择逐日 Raw VS CRPS 的均值最小者；完全同分取较小 gain。不能删除表现差或无效候选后选优。 |
| 解码噪声 | observation_noise=true；seed=2026+export_index；不同 gain/后端使用同日期相同噪声种子。Raw 仅使用原有逆缩放与 [0,1] 裁剪，不新增校准或 ECC。 |
| TEST 准入 | 完整真实 VS 冻结记录须从原始响应重新复核，并与当前协议/准备包绑定；仍需独立显式提交确认。该选择命令不启动 TEST。 |

`quality_status=diagnostic_only` 不等于采样服从目标玻尔兹曼分布。
诊断超限、重复态和相对经典后端的退化都必须如实报告。新规则不要求真机胜过 SA，也不保证真机成功、额度足够或论文结论成立。

## 实现与复核

- `kaiwu_13k_protocol.py`：分离旧硬阈值规则与新诊断模式；统一 gain 排序函数。
- `opsd_vs_selection.py`：只读取 LS/VS 数组，从冻结原始响应重新解码计算 CRPS、Energy、Variogram、Ramp CRPS、PICP90、区间宽度等及平均偏差；拒绝 JSON 自报分数。
- `run_kaiwu_13k_sampling.py`：提交响应后检查 VS 解码有效性；TEST 入口重算同一选择规则，拒绝篡改的分数与 gain。
- `opsd_quality_gate.py`：旧 mock 链不能静默替代新规则验证。
- `tests/test_opsd_vs_selection.py`：临时合成夹具覆盖新规则；不生成论文可用结果。

本地回归命令（不提交真机）：

```powershell
D:\anaconda\envs\wsy\python.exe -m pytest tests/test_kaiwu_13k_protocol.py tests/test_kaiwu_batch_driver.py tests/test_opsd_quality_gate.py tests/test_submit_kaiwu_safety.py tests/test_opsd_quality_validation.py tests/test_opsd_vs_selection.py -q
```

## 下一步顺序

1. 审阅并显式确认新协议；批准的是预先定义的科学决策规则，不是“结果通过”或自动提交授权。
2. 确认后生成独立的新准备目录（例如 v8），并进行零额度完整检查。协议文件任何变更都会改变哈希；不得把 v7 改写哈希后冒充新包，也不得沿用旧 task_name、mock gain freeze。当前旧准备脚本尚未将协议版本传入导出器的 task_name，也不自动生成 `quality_reference_manifest.json`；生成新包之前须接通这两项，不能直接照搬旧准备命令。本轮未扩展修改准备生成器。
3. 新包验证通过后，单独确认一次 100-read VS canary。检查平台返回、扣费方式和工程有效性，再决定是否继续 VS。预算仍为 VS 3,000 + TEST 9,000 reads，canary 满足同一任务身份时计入 VS，不额外重复。
4. 完整真实 VS 返回后，用下面的离线命令冻结 gain。以下为占位说明，当前不能直接执行；它不包含 SDK 提交。

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.opsd_vs_selection --config <已批准且冻结的协议.json> --preparation <新准备目录> --response-audit <真实响应目录\vs_batch_audit.json>
```

5. 人工审阅 VS 预测结果及潜变量诊断，再显式确认 TEST 采样。若发现工程错误，停止排查，不用追加额度掩盖错误。科学指标偏离本身不自动停止，也不自动追加采样。

此修订仅完成代码与规则调整；新准备包验收、实际扣费核验和真机有效性尚未完成。

新模式下，即使 canary 也要求先批准并冻结协议，防止后续批准动作改变哈希、导致已花额度的 canary 不能复用。VS 提交前增加零响应本地预检，先核验模型、数据、VS 条件与参考样本链，再允许发生付费调用。

实际冻结 OPSD 检查点的零额度兼容检查：10 个 VS 条件重建的最大绝对误差为 0；人工构造自旋输入经原模型解码为有限的 `(100,24)` 数组。此项仅检查加载/形状/条件兼容，不是生成质量结果，没有读取 TEST 数组。

## 本次验证记录

- 六个测试文件回归：84 passed，43.17 秒；随后新增的 canary 不得绕过新协议批准测试：1 passed，9.94 秒。合计 85 项通过，均为本地测试。
- 14 条 warnings 来自现有 matplotlib/pyparsing 弃用提示，不是真机响应告警。
- `git diff --check` 通过；仅提示工作树 LF/CRLF 行尾转换。
- 当前新协议 SHA-256：`3e10385b339d1cc8d8be94d11a22afc8c4a0f36fb1df842e377596e274a6d9f6`。
- 历史 v7 记录的协议 SHA-256：`247ae50634bdf0eee8684b58a671b6901615899a82dd0e2325cda3b9d9f72612`。二者不同，不能混用。
- 未提交真机任务、未消耗采样额度、未推送 GitHub，协议批准字段保持 false。

## 2026-10-20 operational addendum

This historical report is retained as written. Current status is recorded in [OPSD current operations](OPSD_CURRENT_OPERATIONS_20261020.md): v8 uses int8-matrix SA for the latent diagnostic reference; floating-point logical SA is a separate backend. The four historical latent thresholds remain diagnostic reference values and are not validated acceptance limits. A complete valid real VS panel selects gain by mean Raw VS CRPS, with exact ties choosing the smaller gain; TEST needs independent explicit confirmation. No real task or TEST metric has been run as part of this addendum.
