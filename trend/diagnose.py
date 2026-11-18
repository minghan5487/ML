"""唯讀診斷：資料新鮮度、每日覆蓋率、breadth 一致性、特徵消融。

不寫入任何檔案、不下載任何資料，只讀 data/ 下既有的 CSV。

用法：
    python -m trend.diagnose              # 全部
    python -m trend.diagnose --no-ablation  # 跳過最花時間的消融實驗
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from trend.config import DATA_DIR, MARKET, STOCKS, VALUATION_DIR
from trend.data import load, load_valuation, path_for
from trend.features import FEATURES, TECHNICAL, build_dataset, market_features, price_features

MARKET_LEVEL = ['mkt_drawdown_250', 'mkt_trend_200', 'breadth_200']
RECENT = 60


def rule(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------- A 價格檔新鮮度
def check_price_freshness():
    rule('A. 價格檔新鮮度')
    rows = []
    for symbol in [MARKET] + [f'{c}.TW' for c in STOCKS]:
        p = path_for(symbol)
        if not p.exists():
            rows.append({'symbol': symbol, 'last_date': None, 'mtime': None, 'rows': 0})
            continue
        df = load(symbol)
        rows.append({
            'symbol': symbol,
            'last_date': df.index[-1].date() if df is not None and len(df) else None,
            'mtime': date.fromtimestamp(p.stat().st_mtime),
            'rows': 0 if df is None else len(df),
        })
    f = pd.DataFrame(rows)
    missing = f[f.last_date.isna()]
    present = f[f.last_date.notna()]
    if present.empty:
        print('沒有任何價格檔，後續診斷無法進行')
        return f

    newest = present.last_date.max()
    stale = present[present.last_date < newest].sort_values('last_date')

    print(f'檔案數 {len(f)}，缺檔 {len(missing)}，最新日期 {newest}')
    print(f'落後最新日期的檔案：{len(stale)} 個')
    if len(stale):
        print(stale[['symbol', 'last_date', 'mtime', 'rows']].to_string(index=False))
        print('\n>>> 這些檔案在近期日期上不會有資料，會直接縮小 breadth_200 的分母。')
    if len(missing):
        print('\n缺檔：', ', '.join(missing.symbol))
    return f


# ---------------------------------------------------------------- B 估值檔健康
def check_valuation_health():
    rule('B. 估值檔健康（M8：空檔會讓下一次讀取崩潰）')
    empty, broken, ok = [], [], 0
    for code in STOCKS:
        p = VALUATION_DIR / f'{code}.csv'
        if not p.exists():
            continue
        if p.stat().st_size <= 2:
            empty.append(code)
            continue
        try:
            v = load_valuation(code)
            if v is None or v.empty:
                empty.append(code)
            else:
                ok += 1
        except Exception as e:
            broken.append((code, type(e).__name__, str(e)[:60]))

    print(f'正常 {ok}，空檔 {len(empty)}，讀取失敗 {len(broken)}')
    if empty:
        print('空檔：', ', '.join(empty))
    if broken:
        for c, t, m in broken:
            print(f'  {c}: {t} {m}')
        print('\n>>> 讀取失敗會讓 build_dataset 直接拋例外，整條自動流程中斷。')
    if empty or broken:
        print('>>> per / pbr 是特徵重要度第 2、3 名，但不在 REQUIRED，')
        print('    所以估值整批失效不會掉樣本，只會被中位數填補、AUC 靜默下滑。')


# ---------------------------------------------------------- C breadth 一致性（核心）
def check_breadth():
    rule('C. breadth_200 一致性與每日覆蓋率（核心檢驗）')

    frames = {}
    for code in STOCKS:
        df = load(f'{code}.TW')
        if df is None or len(df) < 300:
            continue
        frames[code] = price_features(df)

    if not frames:
        print('沒有可用的個股資料')
        return

    above = pd.concat({c: f['faber_trend_200'] > 0 for c, f in frames.items()}, axis=1)
    valid = pd.concat({c: f['faber_trend_200'].notna() for c, f in frames.items()}, axis=1)
    n_valid = valid.sum(axis=1)
    n_above = above.sum(axis=1).astype(float)
    breadth = n_above / n_valid.replace(0, np.nan)

    mkt = market_features()
    tbl = pd.DataFrame({
        'n_valid': n_valid,
        'n_above': n_above.astype(int),
        'breadth_200': breadth.round(3),
        'mkt_drawdown_250': mkt['mkt_drawdown_250'].reindex(breadth.index).round(3),
        'mkt_trend_200': mkt['mkt_trend_200'].reindex(breadth.index).round(3),
    }).dropna(subset=['breadth_200']).tail(RECENT)

    print(f'（最近 {RECENT} 個交易日；股票池共 {len(frames)} 檔）\n')
    print(tbl.to_string())

    last = tbl.iloc[-1]
    print(f'\n最後一日 {tbl.index[-1].date()}：')
    print(f'  分母 n_valid = {int(last.n_valid)} / {len(frames)} 檔')
    print(f'  breadth_200  = {last.breadth_200}  （Bilello 多 的門檻是 > 0.6）')
    print(f'  mkt_drawdown_250 = {last.mkt_drawdown_250}  （Housel 多 的門檻是 < -0.15）')

    if last.n_valid < len(frames) * 0.9:
        print('\n>>> 【命中】分母明顯小於股票池，breadth_200 是用殘缺樣本算的。')
        print('    這能解釋「個股多數跌破 200 日線，breadth 卻 > 0.6」的矛盾。')
    elif last.breadth_200 > 0.6 and last.mkt_drawdown_250 < -0.15:
        print('\n>>> 分母正常，但 breadth > 0.6 與 大盤回檔 > 15% 同時成立。')
        print('    需進一步檢查 ^TWII 與個股 CSV 的日期是否對齊（見 D 節）。')
    else:
        print('\n>>> 未複現矛盾，可能是當日特例或已自行修復。')

    drift = tbl.n_valid.max() - tbl.n_valid.min()
    if drift > 0:
        print(f'\n近 {RECENT} 日分母在 {int(tbl.n_valid.min())} ~ {int(tbl.n_valid.max())} 之間變動（差 {int(drift)} 檔）。')
        print('>>> 這代表標籤基準（若改用橫斷面中位數）會隨下載成敗而變動，須設最少檔數門檻。')


# ---------------------------------------------------------------- D 日期對齊
def check_alignment():
    rule('D. 日期對齊（^TWII vs 個股）')
    m = load(MARKET)
    if m is None:
        print('找不到大盤資料')
        return
    print(f'{MARKET:10s} 最後 5 筆：{[str(d.date()) for d in m.index[-5:]]}')
    for code in list(STOCKS)[:5]:
        s = load(f'{code}.TW')
        if s is None:
            print(f'{code:10s} 缺檔')
            continue
        same = '一致' if list(s.index[-5:]) == list(m.index[-5:]) else '★不一致'
        print(f'{code} {STOCKS[code]:6s} 最後 5 筆：{[str(d.date()) for d in s.index[-5:]]}  {same}')


# ---------------------------------------------------------------- E 每日樣本數
def check_coverage():
    rule('E. 進入訓練的每日樣本數')
    labeled, latest = build_dataset()
    per_day = labeled.groupby(labeled.index).size()
    print(f'labeled 共 {len(labeled):,} 筆，{labeled.index.nunique():,} 個交易日')
    print(f'每日檔數：最小 {per_day.min()}、中位 {int(per_day.median())}、最大 {per_day.max()}')
    thin = per_day[per_day < 20]
    print(f'每日檔數 < 20 的日期：{len(thin)} 天（佔 {len(thin)/len(per_day):.1%}）')
    if len(thin):
        print('  最早 5 天：', [str(d.date()) for d in thin.index[:5]])
        print('  最晚 5 天：', [str(d.date()) for d in thin.index[-5:]])
        print('>>> 改用橫斷面中位數標籤時，這些日期的標籤是純噪音，必須剔除。')
    print(f'\nlatest（預測用）：{len(latest)} 檔，日期 {latest.index.max().date()}')
    if len(latest) < len(STOCKS) * 0.8:
        print('>>> 【命中】latest 檔數明顯不足，但報告文案仍寫「0050 成分股」。')
    print(f'\nM10 檢查：labeled 最後日期 {labeled.index.max().date()}'
          f'（寫進 meta.json 的 data_end）、latest 日期 {latest.index.max().date()}'
          f'（寫進 history.csv 的 data_end）→ 相差 '
          f'{(latest.index.max() - labeled.index.max()).days} 天，同名不同義。')
    return labeled


# ---------------------------------------------------------------- F 特徵消融
def check_ablation(labeled):
    rule('F. 特徵消融：edge 來自個股訊號還是市場層級訊號')
    from trend.train import candidates, evaluate

    sets = {
        'FULL（全部 23 個特徵）': FEATURES,
        'NO_MARKET（去除 3 個市場層級）': [f for f in FEATURES if f not in MARKET_LEVEL],
        'TECHNICAL_ONLY（僅 11 個技術指標）': TECHNICAL,
    }
    print('模型固定用 gradient_boosting，切分與 train.py 相同。\n')
    out = {}
    for name, feats in sets.items():
        r = evaluate(candidates()['gradient_boosting'], labeled, feats)
        out[name] = r
        folds = '  '.join(f'{a:.3f}' for a in r.auc)
        print(f'{name:34s} 平均 AUC {r.auc.mean():.4f}   各折 {folds}')

    full = out['FULL（全部 23 個特徵）'].auc.mean()
    nomkt = out['NO_MARKET（去除 3 個市場層級）'].auc.mean()
    tech = out['TECHNICAL_ONLY（僅 11 個技術指標）'].auc.mean()
    print(f'\n市場層級特徵的邊際貢獻：{full - nomkt:+.4f}')
    print(f'非技術特徵整體的邊際貢獻：{full - tech:+.4f}')
    if full - nomkt > 0.02:
        print('>>> 市場層級特徵確實主導，改用橫斷面標籤有理由。')
    else:
        print('>>> 市場層級特徵的貢獻不大，「模型本質是市場擇時」的說法不成立，')
        print('    改用橫斷面標籤不會解決 edge 衰退，應另尋原因。')


def main():
    print('台股預測模型　唯讀診斷')
    print(f'執行時間 {pd.Timestamp.now():%Y-%m-%d %H:%M}')
    print(f'資料目錄 {DATA_DIR}')

    check_price_freshness()
    check_valuation_health()
    check_breadth()
    check_alignment()
    labeled = check_coverage()

    if '--no-ablation' in sys.argv:
        print('\n（已跳過特徵消融）')
    else:
        check_ablation(labeled)

    print('\n診斷結束，未寫入任何檔案。')


if __name__ == '__main__':
    main()
