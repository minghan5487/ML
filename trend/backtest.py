"""含交易成本的回測：每 HORIZON 個交易日再平衡，買進模型分數前 N 名（等權）。

與兩個基準比較：
  1. 等權買進全部成分股並同樣再平衡（排除選股能力後的報酬）
  2. 等權買進並持有（完全不交易，零成本）

成本假設見 COST_* 常數，預設為台股現行費率。所有報酬皆為扣除成本後的淨值。

用法：
    python -m trend.backtest                 # 用預設參數
    python -m trend.backtest --top 5         # 只買前 5 名
    python -m trend.backtest --fee-discount 1.0   # 不打折的手續費
"""
import sys

import numpy as np
import pandas as pd

from trend.config import HORIZON, REPORT_DIR
from trend.features import FEATURES, build_dataset
from trend.train import N_FOLDS, TEST_DAYS, candidates, walk_forward_splits

COST_FEE_RATE = 0.001425      # 手續費率（買賣各一次）
COST_FEE_DISCOUNT = 0.5       # 券商折扣，0.5 = 五折
COST_TAX_RATE = 0.003         # 證券交易稅（僅賣出）
COST_SLIPPAGE = 0.0010        # 單邊滑價假設
TOP_N = 10


def round_trip_cost(fee_discount=COST_FEE_DISCOUNT):
    """一次完整買進＋賣出的成本（占部位比例）。"""
    fee = COST_FEE_RATE * fee_discount
    return fee * 2 + COST_TAX_RATE + COST_SLIPPAGE * 2


def rebalance_dates(dates):
    """每 HORIZON 個交易日取一個再平衡日，使持有期不重疊。"""
    uniq = np.sort(pd.Index(dates).unique())
    return list(uniq[::HORIZON])


def run(top_n=TOP_N, fee_discount=COST_FEE_DISCOUNT, model_name='gradient_boosting'):
    data, _ = build_dataset()
    cost = round_trip_cost(fee_discount)
    rows = []

    for train_dates, test_dates in walk_forward_splits(data.index):
        train = data[data.index.isin(train_dates)]
        model = candidates()[model_name]
        model.fit(train[FEATURES], train['label'])

        test = data[data.index.isin(test_dates)]
        for d in rebalance_dates(test.index):
            day = test[test.index == d]
            if len(day) < top_n:
                continue
            prob = model.predict_proba(day[FEATURES])[:, 1]
            picked = day.assign(prob=prob).nlargest(top_n, 'prob')

            rows.append({
                'date': d,
                'n_universe': len(day),
                'strategy_gross': picked['future_return'].mean(),
                'equal_weight_gross': day['future_return'].mean(),
                'strategy_net': picked['future_return'].mean() - cost,
                # 基準同樣再平衡 → 同樣付成本；買進持有則完全不付
                'equal_weight_net': day['future_return'].mean() - cost,
                'buy_hold_gross': day['future_return'].mean(),
            })

    if not rows:
        raise RuntimeError('回測沒有產生任何再平衡日')
    return pd.DataFrame(rows).set_index('date').sort_index(), cost


def summarise(bt, cost, top_n, fee_discount):
    periods_per_year = 252 / HORIZON

    def stats(col, pay_cost=True):
        r = bt[col]
        total = (1 + r).prod() - 1
        years = len(r) / periods_per_year
        cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float('nan')
        curve = (1 + r).cumprod()
        mdd = (curve / curve.cummax() - 1).min()
        return {
            '期數': len(r),
            '累積報酬': total,
            '年化報酬': cagr,
            '最大回撤': mdd,
            '單期勝率': (r > 0).mean(),
            '單期平均': r.mean(),
        }

    out = pd.DataFrame({
        f'策略（前{top_n}名，扣成本）': stats('strategy_net'),
        '等權全市場（同樣再平衡，扣成本）': stats('equal_weight_net'),
        '等權買進持有（零成本）': stats('buy_hold_gross'),
        f'策略（前{top_n}名，未扣成本）': stats('strategy_gross'),
    }).T

    lines = [
        '# 含成本回測',
        '',
        f'- 再平衡頻率：每 {HORIZON} 個交易日（持有期不重疊），每年約 {periods_per_year:.1f} 次',
        f'- 選股數：前 {top_n} 名（等權）',
        f'- 單次來回成本：**{cost:.3%}**'
        f'（手續費 {COST_FEE_RATE:.4%}×2×{fee_discount:g} 折 ＋ 證交稅 {COST_TAX_RATE:.1%} ＋ 滑價 {COST_SLIPPAGE:.2%}×2）',
        f'- 年化成本拖累：約 **{cost * periods_per_year:.2%}**',
        '',
        '| 組合 | 期數 | 累積報酬 | 年化報酬 | 最大回撤 | 單期勝率 | 單期平均 |',
        '|---|---|---|---|---|---|---|',
    ]
    for name, r in out.iterrows():
        lines.append(f'| {name} | {int(r["期數"])} | {r["累積報酬"]:+.1%} | {r["年化報酬"]:+.2%} | '
                     f'{r["最大回撤"]:.1%} | {r["單期勝率"]:.1%} | {r["單期平均"]:+.2%} |')

    edge_gross = bt['strategy_gross'].mean() - bt['equal_weight_gross'].mean()
    edge_net = bt['strategy_net'].mean() - bt['buy_hold_gross'].mean()
    lines += [
        '',
        '## 結論',
        '',
        f'- 選股超額報酬（未扣成本，對等權全市場）：**{edge_gross:+.2%}／期**，年化約 {edge_gross * periods_per_year:+.2%}',
        f'- 扣成本後相對「買進持有」的優勢：**{edge_net:+.2%}／期**，年化約 {edge_net * periods_per_year:+.2%}',
        '',
    ]
    if edge_net <= 0:
        lines.append('> **扣除成本後，此策略劣於不做任何交易的等權買進持有。**')
    else:
        lines.append('> 扣除成本後仍有正向優勢，但須再檢視各期穩定度與統計顯著性。')

    lines += [
        '',
        '## 逐期明細（最近 12 期）',
        '',
        '| 再平衡日 | 檔數 | 策略(淨) | 等權(淨) | 買進持有 | 相對買進持有 |',
        '|---|---|---|---|---|---|',
    ]
    for d, r in bt.tail(12).iterrows():
        diff = r['strategy_net'] - r['buy_hold_gross']
        lines.append(f'| {pd.Timestamp(d).date()} | {int(r["n_universe"])} | {r["strategy_net"]:+.2%} | '
                     f'{r["equal_weight_net"]:+.2%} | {r["buy_hold_gross"]:+.2%} | {diff:+.2%} |')

    lines += ['', '> 僅為機器學習練習，不構成投資建議。未計入漲跌停無法成交、流動性不足與停牌等實務限制。']
    return out, '\n'.join(lines) + '\n'


def main():
    argv = sys.argv
    top_n = int(argv[argv.index('--top') + 1]) if '--top' in argv else TOP_N
    disc = float(argv[argv.index('--fee-discount') + 1]) if '--fee-discount' in argv else COST_FEE_DISCOUNT

    bt, cost = run(top_n=top_n, fee_discount=disc)
    table, md = summarise(bt, cost, top_n, disc)

    print(md)
    REPORT_DIR.mkdir(exist_ok=True)
    bt.to_csv(REPORT_DIR / 'backtest.csv')
    (REPORT_DIR / 'backtest.md').write_text(md, encoding='utf-8')
    print(f'已寫出 {REPORT_DIR / "backtest.md"} 與 backtest.csv')


if __name__ == '__main__':
    main()
