"""Finish tables, figures, timestamp sensitivity, and reproducibility report inputs."""
import argparse,hashlib,json,re
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GEFcom2014.neso_wind_rolling import choose_windows
from GEFcom2014.neso_wind_dataset import _forecast_days, _actual_days, _eligibility_rows, _monthly_coverage
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.compare_scenarios import load_scenarios
from GEFcom2014.forecast_quality.neso_wind_results import _scenario_file, _daily_reliability_and_ramp_mae
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator
from GEFcom2014.forecast_quality.neso_paper_checks import OUT,V2,ROOT,NOMINALS,sha,seed_direction_tables,curves


METRIC_DEFINITIONS = '''# Metric definitions

For each date d and quantile level q, compute the empirical coverage c[d,q]
over the 24 hourly cases FIRST. Then take |c[d,q] - q| and average over
quantile levels and dates. This is mean daily reliability MAE, in percentage
points (multiply by 100). It does NOT take an absolute error for each binary
hourly indicator before averaging the hours.

Pooled reliability MAE first computes coverage over ALL test date-hour cases
for each q, then averages |coverage[q] - q| over quantile levels. It uses
q=0.01,...,0.99, the strict comparison target < ensemble quantile, and
percentage points. Each seed is evaluated separately before reporting the
across-seed mean and sample standard deviation. Pooled calibration does not
establish calibration at each date or under every condition.

MAQCE uses q=0.05,0.10,...,0.95 and target <= ensemble quantile, in proportions.
The daily version first averages indicators over 24 hours for each date and q,
then takes absolute coverage errors and averages levels/dates. The pooled
version first averages indicators over all test date-hour cases for each q.
Annual pooled errors must be recomputed from coverage counts, not averaged
from quarterly absolute errors.

Coverage and width use central equal-tailed intervals at 50%, 80%, 90%, 95%;
width is MW. Calibration is fitted on each window's VS only. Cal+ECC preserves
Cal hourly marginals. Mean daily metrics remain in the frozen results;
CRPS, Energy, Variogram, Ramp CRPS and the trained models are unchanged.
'''


def forecast_with_timezone(raw, timezone):
    """Change only naive issue-time interpretation; run the actual production filter."""
    if timezone not in ('UTC', 'Europe/London'):
        raise ValueError('Unsupported issue timezone')
    frame = raw.copy()
    strings = frame.Forecast_Timestamp.astype('string')
    aware = strings.str.contains(r'(?:Z|[+-]\d\d:?\d\d)$', case=False, regex=True, na=False)
    normalized = pd.to_datetime(strings, utc=True, errors='coerce', format='mixed')
    if timezone == 'Europe/London':
        local = pd.to_datetime(strings[~aware], errors='coerce', format='mixed')
        normalized.loc[~aware] = local.dt.tz_localize(
            'Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    frame['Forecast_Timestamp'] = normalized.map(lambda t: t.isoformat() if pd.notna(t) else '')
    audit = {}
    with TemporaryDirectory(prefix='neso_timestamp_check_') as temp:
        path = Path(temp) / 'forecast.csv'
        frame.to_csv(path, index=False)
        forecast, reasons, _ = _forecast_days(path, audit=audit)
    return forecast, audit, reasons


def timezone_date_differences(utc_eligibility, local_eligibility):
    """Compare complete calendar-date eligibility records, including exclusions."""
    left = utc_eligibility.set_index('date')
    right = local_eligibility.set_index('date')
    dates = sorted(set(left.index) | set(right.index))
    rows = []
    for date in dates:
        utc = left.loc[date] if date in left.index else None
        local = right.loc[date] if date in right.index else None
        eligible_utc = utc is not None and utc.status == 'eligible'
        eligible_local = local is not None and local.status == 'eligible'
        rows.append({
            'date': date,
            'eligible_utc': eligible_utc,
            'eligible_london': eligible_local,
            'utc_reason': '' if utc is None else utc.reason,
            'london_reason': '' if local is None else local.reason,
            'difference': ('both' if eligible_utc and eligible_local else
                           'UTC_only' if eligible_utc else
                           'London_only' if eligible_local else 'neither'),
        })
    return pd.DataFrame(rows)


def strict_timezone_sensitivity():
    source = V2/'sources/historic_day_ahead_wind_forecasts_20261022.csv'
    raw = pd.read_csv(source, dtype={'Forecast_Timestamp':'string'})
    meta = json.loads((V2/'data/neso_wind_daily.json').read_text(encoding='utf-8'))
    actual_paths = []
    for item in meta['source_files']:
        if Path(item['path']).name.startswith('monthly-operational-metered'):
            path = ROOT/'datasets/neso'/Path(item['path']).name
            if sha(path) != item['sha256']:
                raise ValueError('Actual source changed: ' + str(path))
            actual_paths.append(path)
    selected, frames, counts, eligibility_by_timezone = {}, [], [], {}
    frozen_arrays, _, frozen_dates = load_daily_bundle(V2/'data/neso_wind_daily.npz')
    frozen_by_date = {}
    for split in ('ls', 'vs', 'test'):
        for i, date in enumerate(frozen_dates[split]):
            frozen_by_date[date] = (frozen_arrays['x_' + split][i, :24],
                                     frozen_arrays['y_' + split][i])
    frozen_forecast_errors, frozen_actual_errors = [], []
    for timezone in ('UTC', 'Europe/London'):
        forecast, audit, _ = forecast_with_timezone(raw, timezone)
        actual_audit = {}
        actual, _, _, _ = _actual_days(actual_paths, audit=actual_audit)
        frame = _eligibility_rows(audit, actual_audit, forecast, actual)
        eligibility_by_timezone[timezone] = frame
        selected[timezone] = set(frame.loc[frame.status == 'eligible', 'date'])
        monthly = _monthly_coverage(frame).set_index('month')
        all_months = pd.period_range(frame.month.min(), frame.month.max(), freq='M').astype(str)
        monthly = monthly.reindex(all_months, fill_value=0).rename_axis('month').reset_index().fillna(0)
        monthly.insert(0, 'interpretation', timezone)
        counts.append(monthly)
        frame.insert(0, 'interpretation', timezone)
        frames.append(frame)
        if timezone == 'UTC':
            if selected[timezone] != set(frozen_by_date):
                raise ValueError('UTC eligible dates differ from frozen bundle dates')
            for date, (frozen_x, frozen_y) in frozen_by_date.items():
                day = __import__('datetime').date.fromisoformat(date)
                if day not in forecast or day not in actual:
                    raise ValueError('Frozen bundle date missing from freshly cleaned sources: ' + date)
                frozen_forecast_errors.append(float(np.max(np.abs(frozen_x - forecast[day]['values']))))
                frozen_actual_errors.append(float(np.max(np.abs(frozen_y - actual[day]))))
    if max(frozen_forecast_errors, default=0.) > 1e-4:
        raise ValueError('UTC cleaned forecast values differ from frozen bundle')
    if max(frozen_actual_errors, default=0.) > 1e-3:
        raise ValueError('UTC cleaned actual values differ from frozen bundle beyond float32 tolerance')
    frozen = pd.read_csv(V2/'date_eligibility.csv')
    expected = set(frozen.loc[frozen.status == 'eligible', 'date'])
    if selected['UTC'] != expected:
        raise ValueError('Full UTC cleaning differs from frozen eligibility; investigate source drift')
    pd.concat(frames, ignore_index=True).to_csv(OUT/'timezone_eligibility.csv', index=False)
    pd.concat(counts, ignore_index=True).fillna(0).to_csv(OUT/'timezone_sensitivity_monthly.csv', index=False)
    utc, local = selected['UTC'], selected['Europe/London']
    both = utc & local
    timezone_date_differences(eligibility_by_timezone['UTC'],
                              eligibility_by_timezone['Europe/London']).to_csv(
                                  OUT/'timezone_date_differences.csv', index=False)
    try:
        windows, _ = choose_windows(sorted(local))
        candidates = [{'test_start':w['test_start'], 'test_end':w['test_end'],
                       'LS':len(w['ls']), 'VS':len(w['vs']), 'TEST':len(w['test'])} for w in windows]
    except ValueError:
        candidates = []
    old = json.loads((V2.parent/'fa_bm_vae_neso_wind_classical_v1/data/neso_wind_daily.json').read_text(encoding='utf-8'))
    old_dates = old['dates_by_split']
    old_vs, old_test = set(old_dates['vs']), set(old_dates['test'])
    impact = {
        'judgment':'UNRESOLVED',
        'formal_assumption':'Naive Forecast_Timestamp is interpreted as UTC; timezone and publication/vintage semantics remain unverified. Local interpretation is diagnostic only.',
        'validation':'Both interpretations use production _forecast_days and _actual_days; only naive issue timestamp interpretation changes.',
        'utc_eligible_days':len(utc), 'london_eligible_days':len(local),
        'common_dates':len(both), 'utc_only':len(utc-local), 'london_only':len(local-utc),
        'utc_matches_frozen_eligibility':True, 'existing_rolling_windows_unchanged':True,
        'utc_matches_frozen_bundle_values':True,
        'max_abs_forecast_difference_mw':max(frozen_forecast_errors, default=0.),
        'max_abs_actual_difference_mw':max(frozen_actual_errors, default=0.),
        'actual_cleaning_repeated_for_each_timezone':True,
        'local_candidate_quarter_windows':candidates,
        'old_90d_vs_dates_missing_under_local':len(old_vs-local),
        'old_90d_test_dates_missing_under_local':len(old_test-local),
        'old_90d_vs_interval_added_dates':sum(min(old_vs)<=d<=max(old_vs) for d in local-utc),
        'old_90d_test_interval_added_dates':sum(min(old_test)<=d<=max(old_test) for d in local-utc),
        'actual_data_and_targets_held_fixed':True,
        'retrain_recommendation':'No retraining now. If official evidence changes the interpretation and inputs, rebuild affected splits/scalers/graphs and retrain; do not relabel old checkpoints.'}
    (OUT/'impact_on_existing_experiments.json').write_text(json.dumps(impact, indent=2), encoding='utf-8')
    return impact, [source, V2/'data/neso_wind_daily.json', V2/'date_eligibility.csv', *actual_paths]


def verify_fa_reliability_from_saved_scenarios():
    """Rebuild the annual FA Cal reliability values from archived VS/TEST scenes."""
    stored = pd.read_csv(OUT/'calibration_metrics_per_seed.csv')
    daily_metrics_path = V2/'rolling_results/metrics_daily.csv'
    saved_daily = pd.read_csv(daily_metrics_path)
    rows, inputs = [], []
    for seed in (0, 1, 2):
        samples, targets, recomputed_daily = [], [], []
        for wi in range(1, 5):
            home = V2/'rolling'/f'quarter_{wi}'
            bundle = home/'data/neso_wind_daily.npz'
            arrays, _, dates = load_daily_bundle(bundle)
            yvs, ytest = arrays['y_vs'].reshape(-1), arrays['y_test'].reshape(-1)
            vs_path = _scenario_file(home/'formal', 'fa_bm_vae', seed, 'VS', 100)
            test_path = _scenario_file(home/'formal', 'fa_bm_vae', seed, 'TEST', 100)
            raw_vs = load_scenarios(vs_path, len(yvs))
            raw_test = load_scenarios(test_path, len(ytest))
            calibrator = HourlyAffineCalibrator(limits=(0., float('inf'))).fit(raw_vs, yvs)
            cal_test = calibrator.transform(raw_test).reshape(-1, 24, 100)
            ytest_matrix = ytest.reshape(-1, 24)
            samples.append(cal_test)
            targets.append(ytest_matrix)
            daily_rel, _ = _daily_reliability_and_ramp_mae(
                cal_test.reshape(-1, 100), ytest_matrix)
            actual_daily = saved_daily[(saved_daily.window == f'quarter_{wi}') &
                                       (saved_daily.model == 'FA-BM-VAE') &
                                       (saved_daily.seed == seed) &
                                       (saved_daily.variant == 'Cal')].sort_values('date')
            if actual_daily.date.astype(str).tolist() != list(dates['test']):
                raise ValueError('Daily reliability date grid differs from frozen TEST dates')
            if not np.allclose(daily_rel, actual_daily.Reliability_MAE_pp.to_numpy(),
                               atol=1e-12, rtol=0):
                raise ValueError('Recomputed daily reliability differs from saved daily metrics')
            recomputed_daily.extend(daily_rel.tolist())
            row = stored[(stored.window == f'quarter_{wi}') & (stored.model == 'FA-BM-VAE') &
                         (stored.seed == seed) & (stored.variant == 'Cal')]
            if len(row) != 1:
                raise ValueError('Missing or duplicate stored daily reliability summary')
            inputs.extend([bundle, vs_path, test_path])
        pooled, _, _, _ = curves(np.concatenate(samples), np.concatenate(targets))
        mean_daily = float(np.mean(recomputed_daily))
        expected = stored[(stored.window == 'FULL_YEAR') & (stored.model == 'FA-BM-VAE') &
                          (stored.seed == seed) & (stored.variant == 'Cal')]
        if len(expected) != 1 or not np.isclose(pooled, expected.iloc[0].pooled_reliability_mae_pp,
                                                atol=1e-12, rtol=0) or not np.isclose(
                mean_daily, expected.iloc[0].mean_daily_reliability_mae_pp, atol=1e-12, rtol=0):
            raise ValueError('Saved annual FA reliability values do not reproduce')
        rows.append({'seed':seed, 'pooled_reliability_mae_pp_recomputed':pooled,
                     'pooled_reliability_mae_pp_saved':float(expected.iloc[0].pooled_reliability_mae_pp),
                     'mean_daily_reliability_mae_pp_recomputed':mean_daily,
                     'mean_daily_reliability_mae_pp_saved':float(expected.iloc[0].mean_daily_reliability_mae_pp),
                     'n_test_dates':len(recomputed_daily)})
    inputs.append(daily_metrics_path)
    result = {'method':'saved VS scenes fit the frozen hourly affine calibrator; saved TEST scenarios and targets; 99 levels; y < empirical quantile',
              'training_executed':False, 'new_scenarios_generated':False,
              'rows':rows, 'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))}}
    (OUT/'reliability_recomputation_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def correct_reports():
    """Repair supplementary outputs only; do not rerun training or bootstrap."""
    backup = OUT.with_name('paper_checks_v1_pre_correction_20261023')
    previous = json.loads((backup/'analysis_identity.json').read_text(encoding='utf-8'))
    # These three supplementary inputs become revised outputs. Validate their
    # archived originals, while continuing to validate all untouched live inputs.
    replaced = {OUT/name for name in ('calibration_metrics_summary.csv',
                                      'paired_differences_by_seed.csv',
                                      'timezone_date_differences.csv', 'timezone_sensitivity_monthly.csv')}
    retained_inputs, archived_inputs = {}, {}
    for rel, expected in previous['input_sha256'].items():
        path = ROOT/rel
        if path in replaced:
            path = backup/path.name
            archived_inputs[str(path.relative_to(ROOT))] = expected
        else:
            retained_inputs[rel] = expected
        if sha(path) != expected:
            raise ValueError('Frozen input changed: ' + rel)
    protocol = json.loads((V2/'protocol_v2.json').read_text(encoding='utf-8'))
    counts = {w['name']:len(w['dates']['test']) for w in protocol['windows']}
    quarterly = pd.read_csv(backup/'paired_differences_by_seed.csv')
    differences, directions = seed_direction_tables(quarterly, counts)
    # Cross-check all weighted annual differences against the original per-date data.
    daily_path = V2/'rolling_results/metrics_daily.csv'
    daily = pd.read_csv(daily_path)
    marginal_columns = ['Reliability_MAE_pp', 'MAQCE', 'PICP90', 'interval_width90_MW']
    cal = daily[daily.variant == 'Cal'].set_index(['window','model','seed','date'])[marginal_columns].sort_index()
    ecc = daily[daily.variant == 'Cal+ECC'].set_index(['window','model','seed','date'])[marginal_columns].sort_index()
    if not cal.index.equals(ecc.index) or not np.array_equal(cal.to_numpy(), ecc.to_numpy()):
        raise ValueError('Saved Cal and Cal+ECC marginal metrics differ')
    for row in differences[differences.window == 'FULL_YEAR'].itertuples():
        part = daily[(daily.seed == row.seed) & (daily.variant == row.variant)]
        left = part[part.model == row.model].set_index('date')[row.metric].sort_index()
        right = part[part.model == row.comparator].set_index('date')[row.metric].sort_index()
        if left.index.has_duplicates or not left.index.equals(right.index) or len(left) != sum(counts.values()):
            raise ValueError('Incomplete annual paired dates')
        if not np.isclose((left-right).mean(), row.mean_difference, atol=1e-10, rtol=0):
            raise ValueError('Annual seed difference disagrees with direct daily aggregation')
    differences.to_csv(OUT/'paired_differences_by_seed.csv', index=False)
    directions.to_csv(OUT/'seed_direction_summary.csv', index=False)
    impact, extra_inputs = strict_timezone_sensitivity()
    reliability_check = verify_fa_reliability_from_saved_scenarios()
    (OUT/'METRIC_DEFINITIONS.md').write_text(METRIC_DEFINITIONS, encoding='utf-8')
    annual = directions[directions.window == 'FULL_YEAR']
    seasonal = directions[directions.window != 'FULL_YEAR']
    main = differences[(differences.window == 'FULL_YEAR') & (differences.variant == 'Cal') & (differences.metric == 'CRPS_MW')]
    rows = ['| Comparator | seed 0 | seed 1 | seed 2 |', '|---|---:|---:|---:|']
    for model, part in main.groupby('comparator', sort=False):
        values = part.set_index('seed').mean_difference
        rows.append('| ' + model + ' | ' + ' | '.join(f'{values[s]:.4f}' for s in (0,1,2)) + ' |')
    seed_note = (f'全年{len(annual)}个比较/指标组合中，{int(annual.mixed_seed_signs.sum())}个存在seed方向差异；'
                 f'分窗口的{len(seasonal)}个组合中，{int(seasonal.mixed_seed_signs.sum())}个存在seed方向差异。'
                 '二者均不能用跨季节符号翻转来代替。五个主基线的全年Cal CRPS比较在三个seed下均为负；'
                 'J=0对照存在seed方向差异。seed方向一致不等于日期配对置信区间排除零。')
    report = f'''# NESO论文结果核查（修正版）

本轮只修正分析代码、补充CSV和报告。训练、场景、冻结协议和原始主表未改变。
旧分析及旧源码已保存在 `{backup.name}`，来源记录见 correction_identity.json。

## 可靠性指标

逐日Reliability MAE先在每个日期、每个分位水平上对24小时的二值指示求平均，
再取覆盖率绝对误差，最后平均分位水平与日期。不能先对每小时指示取绝对误差。
全年pooled指标先汇总全部日期小时再取绝对误差。FA-BM-VAE Cal的三个seed
pooled值为2.6826、2.6987、3.1120 pp，均值2.8311 pp；mean-daily均值为20.1308 pp。
这些指标的数值未修改，修正的是定义表述。pooled校准不等于逐日条件校准。

## Seed稳定性

{seed_note}

差值单位为MW，定义为FA-BM-VAE减对照。全年按实际日期数加权，并与原逐日数据直接汇总核对。

{chr(10).join(rows)}

## 日期统计

3/7/14天块长的原敏感性分析未重算：Cal CRPS对Gaussian和residual bootstrap的区间均低于零；
对GMM、Score-SDE、Spline Flow及J=0的区间均包含零。区间条件于当前已训练模型，
不能当作完整训练随机性区间，也未进行多重比较控制。

## 时间戳

判断仍为UNRESOLVED，没有新增能确定历史时区或真实发布时间的官方证据。
两种解释现在都复用正式_forecast_days和_actual_days的完整清洗，并保存逐日排除原因。
UTC得到{impact['utc_eligible_days']}天，London当地时间解释得到{impact['london_eligible_days']}天，
其中London-only为{impact['london_only']}天；UTC集合与冻结数据完全一致。
原90天VS的日历范围内新增{impact['old_90d_vs_interval_added_dates']}个候选日期，
原TEST日历范围内新增{impact['old_90d_test_interval_added_dates']}天。
这不同于“原来保留的90天是否仍然有效”；不能因保留日期未丢失就声称验证期没有新增日期。
严格清洗通过不等于时区解释已证实。候选窗口仅用于诊断，不替换冻结窗口。
本轮不启动训练；只有官方证据改变解释且影响输入时才规划受影响模型的重训。

## 重现

`D:/anaconda/envs/wsy/python.exe -m GEFcom2014.forecast_quality.neso_paper_checks_finalize --corrections-only`

该命令重建本次修正输出，校验{len(retained_inputs)}个未修改输入和{len(archived_inputs)}个已归档的原分析输入；不重训、不重跑bootstrap。
'''
    (OUT/'PAPER_CHECKS_REPORT.md').write_text(report, encoding='utf-8')
    original_stats = (backup/'STATISTICAL_ROBUSTNESS.md').read_text(encoding='utf-8')
    (OUT/'STATISTICAL_ROBUSTNESS.md').write_text(original_stats + '\n## Corrected seed summary\n\n' + seed_note + '\n\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    original_audit = (backup/'TIMESTAMP_AUDIT.md').read_text(encoding='utf-8')
    if impact['utc_eligible_days'] != 2104 or impact['london_eligible_days'] != 2361:
        original_audit = original_audit.split('Sensitivity held target times')[0]
    (OUT/'TIMESTAMP_AUDIT.md').write_text(original_audit + f'\n## Full-filter correction\n\nBoth interpretations now call the unchanged production _forecast_days and _actual_days. Finite/nonnegative values, settlement periods, local date mapping, exact contiguous half-hour target times, DST and actual component totals are checked. UTC={impact["utc_eligible_days"]}; London={impact["london_eligible_days"]}; local-only={impact["london_only"]}. UTC matches the frozen eligibility set. Detailed reasons: timezone_eligibility.csv. This does not resolve timezone semantics or verify publication vintage.\n', encoding='utf-8')
    reliability = pd.read_csv(OUT/'calibration_metrics_per_seed.csv')
    fields = [c for c in reliability.columns if c.endswith('_pp') or c.endswith('_maqce')
              or c.startswith('coverage_') or c.startswith('width_')]
    summary = reliability.groupby(['window','model','variant']).agg(
        {c:['mean','std'] for c in fields}).reset_index()
    summary.to_csv(OUT/'calibration_metrics_summary.csv', index=False)
    fa = reliability[(reliability.model == 'FA-BM-VAE') &
                     (reliability.variant == 'Cal') &
                     (reliability.window == 'FULL_YEAR')].sort_values('seed')
    if list(fa.seed.astype(int)) != [0, 1, 2]:
        raise ValueError('Full-year FA-BM-VAE reliability seed rows are incomplete')
    reliability_values = ', '.join(
        f"seed {r['seed']}: {r['pooled_reliability_mae_pp_recomputed']:.4f} pp pooled / "
        f"{r['mean_daily_reliability_mae_pp_recomputed']:.4f} pp mean-daily"
        for r in reliability_check['rows'])
    review_text = f'''# NESO 修正版独立复核

本次只核验已有四窗口模型、场景和逐日结果；未训练模型、未更改划分或模型，未运行 bootstrap 重采样，也未触及 OPSD 真机主线。

## 全年 seed 差值

全年 FA-BM-VAE 减对照模型的差值，先在每个 seed 内按共同日期逐日相减，再对该年度实际的 361 个测试日期求均值。没有对四个季度均值作等权平均。结果见 `paired_differences_by_seed.csv` 的 `FULL_YEAR` 行。

五个主基线的 Cal CRPS 在 seed 0/1/2 下均为负，表示 FA-BM-VAE 的 CRPS 较低。J=0 的 Cal CRPS 差值分别约为 −2.8565、−1.9313、+2.7029 MW；第三个 seed 方向相反。13 个全年比较/指标组合中，7 个组合的 seed 差值正负方向不一致；52 个分窗口组合中，23 个存在 seed 方向不一致。季节间方向变化与同一窗口内 seed 差异分开统计，seed 不是额外测试日期。

| 对照模型（Cal CRPS） | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
{chr(10).join(rows[2:])}

## 可靠性指标定义

逐日 Reliability MAE 对每个日期、每个分位数先统计 24 小时覆盖率，再取绝对误差并对 99 个分位数及日期求平均；采用严格比较 `y < quantile`，单位为百分点。pooled Reliability MAE 先合并所选范围内所有日期和小时的覆盖计数，再对同一 99 点网格的绝对覆盖误差求均值。两者聚合顺序不同，不能互换。本轮从存档 VS/TEST 场景重新拟合冻结校准器并复算全年指标，逐 seed 与保存结果在 1e-12 pp 容差内一致：{reliability_values}。三 seed pooled 均值为 2.8311 pp，mean-daily 均值为 20.1308 pp。详细输入哈希见 `reliability_recomputation_check.json`。

MAQCE 使用 0.05 至 0.95、步长 0.05 的 19 个分位数网格，比较符号为 `y <= quantile`；pooled 版本先汇总日期和小时覆盖率。现有 Cal+ECC 与 Cal 的逐日 Reliability MAE、MAQCE、PICP90 和区间宽度在 {len(cal)} 个配对模型/seed/日期单元中完全一致，因此边际可靠性复用 Cal 结果有逐日证据。该结果不证明每个日期或条件下校准良好，也不改变 CRPS 等原指标。

## 时区与完整清洗敏感性

判定仍为 **UNRESOLVED**。两种解释都经过生产版 `_forecast_days` 和 `_actual_days` 完整校验：有效值、48 个 settlement period、UK 日期映射、连续半小时区间、DST、实际值总量与分项一致性，以及逐日排除原因。actual 清洗针对每种解释独立重跑；目标与实际出力保持不变。逐日明细包括不合格日期，月表保留零覆盖月份。

UTC 解释得到 2104 个日期，与冻结 bundle 日期及逐日输入值一致。London 当地时间解释得到 2361 个日期，其中 257 个仅在该解释下合格。UTC 重建预测与冻结 bundle 的最大绝对误差为 {impact['max_abs_forecast_difference_mw']:.8g} MW；实际值最大误差为 {impact['max_abs_actual_difference_mw']:.8g} MW（bundle 使用 float32）。原冻结 VS/TEST 日期在两种解释下均仍可用；原 VS 日历范围内另有 206 个 London 候选日期，TEST 日历范围内新增 0 个。不能用新增候选日期替换既有 checkpoint 输入，也不能据此说原 VS 选择不受候选集合差异影响。

官方字段说明尚不能证明历史 `Forecast_Timestamp` 的时区、真实发布时间语义及归档版本历史。现有实验成立依赖“无偏移时间戳按 UTC 解释，并采用冻结 cutoff”这一假设；覆盖日期多少不能裁定正确解释。若官方证据推翻该假设，相关数据、缩放、图结构及依赖这些输入的模型需要按新冻结划分重建；不能只给旧 checkpoint 更换数据窗口。

## 来源与验证

旧检查备份位于 `paper_checks_v1_pre_correction_20261023/`。`correction_identity.json` 验证了冻结输入、归档的旧分析输入及修订输出哈希，并记录 `training_executed=false`、`bootstrap_recomputed=false`。`analysis_identity.json` 与 `review_identity.json` 记录源码、输入及报告指纹。四窗口训练、场景和 bootstrap 结果均复用，未伪称本轮重新生成。

针对性测试：`D:/anaconda/envs/wsy/python.exe -m pytest tests/test_neso_paper_checks.py tests/test_neso_wind_dataset.py -q`，本轮 10 项通过。官方时间语义仍未决。
'''
    (OUT/'REVIEW_AFTER_CORRECTION.md').write_text(review_text, encoding='utf-8')
    (OUT/'PAPER_CHECKS_REPORT.md').write_text(review_text, encoding='utf-8')
    correction_inputs = dict(archived_inputs)
    correction_inputs.update({str(p.relative_to(ROOT)):sha(p) for p in [daily_path, *extra_inputs]})
    correction_inputs.update(reliability_check['input_sha256'])
    identity = {
        'revision':'paper-checks-correction-20261023',
        'previous_identity':str((backup/'analysis_identity.json').relative_to(ROOT)),
        'previous_identity_sha256':sha(backup/'analysis_identity.json'),
        'retained_inputs_verified_unchanged':len(retained_inputs),
        'archived_replaced_inputs_verified':len(archived_inputs),
        'bootstrap_recomputed':False, 'training_executed':False,
        'cal_ecc_marginal_metrics_exactly_equal':True,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'GEFcom2014/forecast_quality/neso_paper_checks.py', ROOT/'GEFcom2014/neso_wind_dataset.py', ROOT/'GEFcom2014/neso_wind_rolling.py', ROOT/'tests/test_neso_paper_checks.py']},
        'input_sha256':correction_inputs,
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in (
            'analysis_identity.json','correction_identity.json','review_identity.json')}}
    snapshot_paths = [backup/'source_snapshot/GEFcom2014/forecast_quality/neso_paper_checks.py',
                      backup/'source_snapshot/GEFcom2014/forecast_quality/neso_paper_checks_finalize.py',
                      backup/'source_snapshot/tests/test_neso_paper_checks.py']
    review_identity = {
        'revision':'neso-independent-review-after-correction-20261023',
        'previous_analysis_identity_sha256':identity['previous_identity_sha256'],
        'correction_code_sha256':identity['code_sha256'],
        'input_sha256':identity['input_sha256'],
        'output_sha256':identity['output_sha256'],
        'pre_correction_source_snapshot_sha256':{str(p.relative_to(ROOT)):sha(p) for p in snapshot_paths},
        'training_executed':False, 'bootstrap_recomputed':False,
        'frozen_model_split_and_protocol_changed':False}
    (OUT/'review_identity.json').write_text(json.dumps(review_identity, indent=2), encoding='utf-8')
    identity['output_sha256']['review_identity.json'] = sha(OUT/'review_identity.json')
    (OUT/'correction_identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    current_identity = dict(previous)
    current_identity.update({
        'revision':identity['revision'],
        'previous_identity':identity['previous_identity'],
        'previous_identity_sha256':identity['previous_identity_sha256'],
        'code_sha256':identity['code_sha256'],
        'input_sha256':dict(retained_inputs, **identity['input_sha256']),
        'outputs':dict(identity['output_sha256'], **{'correction_identity.json':sha(OUT/'correction_identity.json')}),
        'artifact_provenance':'Calibration per-seed metrics and bootstrap outputs are reused from the archived original identity. This revision regenerated date-weighted seed summaries, complete timezone diagnostics, the calibration summary, reliability verification, and documentation from frozen inputs.'})
    (OUT/'analysis_identity.json').write_text(json.dumps(current_identity, indent=2), encoding='utf-8')
    return {'annual_mixed_seed_signs':int(annual.mixed_seed_signs.sum()),
            'window_mixed_seed_signs':int(seasonal.mixed_seed_signs.sum()), 'timezone':impact}

def main():
 f=pd.read_csv(OUT/'calibration_metrics_per_seed.csv')
 # CSV first block has the named header, annual rows were appended in same schema.
 full=f[f.window=='FULL_YEAR']; summary=full.groupby(['model','variant']).agg({c:['mean','std'] for c in full.columns if c.endswith('_pp') or c.endswith('_maqce') or c.startswith('coverage_') or c.startswith('width_')}).reset_index()
 summary.to_csv(OUT/'calibration_metrics_summary.csv',index=False)
 lines=['# Reliability and coverage, full year','', '| Model | Variant | Daily Rel MAE (pp) | Pooled Rel MAE (pp) | Daily MAQCE | Pooled MAQCE |','|---|---|---:|---:|---:|---:|']
 for _,r in summary.iterrows():
  def fmt(metric):
   m=float(r[(metric,'mean')]); s=float(r[(metric,'std')]); return f'{m:.4f} ± {s:.4f}'
  lines.append(f"| {r[('model','')]} | {r[('variant','')]} | {fmt('mean_daily_reliability_mae_pp')} | {fmt('pooled_reliability_mae_pp')} | {fmt('mean_daily_maqce')} | {fmt('pooled_maqce')} |")
 (OUT/'reliability_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'reliability_table.tex').write_text('\\begin{tabular}{llrrrr}\nModel & Variant & Daily reliability MAE (pp) & Pooled reliability MAE (pp) & Daily MAQCE & Pooled MAQCE \\\\\n\\midrule\n'+'\n'.join(f"{r[('model','')]} & {r[('variant','')]} & {float(r[('mean_daily_reliability_mae_pp','mean')]):.4f} $\\pm$ {float(r[('mean_daily_reliability_mae_pp','std')]):.4f} & {float(r[('pooled_reliability_mae_pp','mean')]):.4f} $\\pm$ {float(r[('pooled_reliability_mae_pp','std')]):.4f} & {float(r[('mean_daily_maqce','mean')]):.4f} & {float(r[('pooled_maqce','mean')]):.4f} \\\\" for _,r in summary.iterrows())+'\n\\end{tabular}\n',encoding='utf-8')
 # Reliability and interval width curves by pooled variants and season/window.
 cov=pd.read_csv(V2/'rolling_results/coverage_by_nominal.csv')
 for label,sub in [('full_year',cov),*[(f'quarter_{i}',cov[cov.window==f'quarter_{i}']) for i in range(1,5)]]:
  fig,axs=plt.subplots(1,2,figsize=(12,4.5))
  for (model,var),g in sub.groupby(['model','variant']):
   g=g.groupby('nominal_coverage').apply(lambda x: np.average(x.actual_coverage,weights=x.n_dates),include_groups=False)
   axs[0].plot(g.index,g.values,marker='o',label=f'{model} / {var}')
  axs[0].plot([.5,.95],[.5,.95],'k--'); axs[0].set(xlabel='Nominal coverage',ylabel='Observed coverage',title=label)
  for (model,var),g in sub.groupby(['model','variant']):
   g=g.groupby('nominal_coverage').apply(lambda x: np.average(x.mean_interval_width_mw,weights=x.n_dates),include_groups=False)
   axs[1].plot(g.index,g.values,marker='o',label=f'{model} / {var}')
  axs[1].set(xlabel='Nominal coverage',ylabel='Mean interval width (MW)',title=label)
  axs[0].legend(fontsize=5); axs[1].legend(fontsize=5); fig.tight_layout(); fig.savefig(OUT/f'reliability_coverage_width_{label}.png',dpi=170); plt.close(fig)
 return correct_reports()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrections-only', action='store_true',
                        help='Repair seed summaries, definitions and timezone filtering without training or bootstrap')
    args = parser.parse_args()
    print(json.dumps(correct_reports() if args.corrections_only else main(), indent=2))
