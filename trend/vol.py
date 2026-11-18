"""波動度方向模型：預測未來 HORIZON 日的實現波動是否高於過去 HORIZON 日。

為什麼做這個：價格方向幾乎不可預測（本專案實測 AUC ≈ 0.55、準確率低於多數類基準），
但波動度有很強的自相關與群聚性（volatility clustering），可預測性高出一個量級。
同一套資料、同一套 walk-forward 與 embargo，只是換一個標籤。

用法：
    python -m trend.vol
"""
import numpy as np
import pandas as pd

from trend.config import HORIZON, REPORT_DIR
from trend.features import FEATURES, build_dataset
from trend.train import candidates, walk_forward_splits
from sklearn.metrics import accuracy_score, roc_auc_score

LABEL = 'label_vol'


def evaluate_vol(model, data, features):
    scores = []
    for train_dates, test_dates in walk_forward_splits(data.index):
        train = data[data.index.isin(train_dates)]
        test = data[data.index.isin(test_dates)]
        if train.empty or test.empty:
            continue

        model.fit(train[features], train[LABEL])
        prob = model.predict_proba(test[features])[:, 1]
        majority = int(train[LABEL].mean() >= 0.5)

        scores.append({
            'period': f'{pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()}',
            'auc': roc_auc_score(test[LABEL], prob),
            'accuracy': accuracy_score(test[LABEL], prob >= 0.5),
            'baseline': accuracy_score(test[LABEL], np.full(len(test), majority)),
            'positive_rate': test[LABEL].mean(),
            # 只看模型最有把握的一成，準確率通常明顯更高
            'top_decile_accuracy': accuracy_score(
                test[LABEL][prob >= np.quantile(prob, 0.9)],
                np.ones((prob >= np.quantile(prob, 0.9)).sum())),
        })
    return pd.DataFrame(scores)


def main():
    data, _ = build_dataset()
    data = data[data[LABEL].notna()]
    print(f'資料 {len(data):,} 筆，{data.index.nunique():,} 個交易日，'
          f'正樣本率 {data[LABEL].mean():.1%}')

    results = {}
    for name, model in candidates().items():
        f = evaluate_vol(model, data, FEATURES)
        results[name] = f
        print(f'{name:18s} AUC {f.auc.mean():.4f}  準確率 {f.accuracy.mean():.2%}  '
              f'基準 {f.baseline.mean():.2%}  前10%把握 {f.top_decile_accuracy.mean():.2%}')

    best_name = max(results, key=lambda n: results[n].auc.mean())
    best = results[best_name]

    lines = [
        f'# 波動度方向模型（未來 {HORIZON} 日實現波動 vs 過去 {HORIZON} 日）',
        '',
        '價格方向幾乎不可預測；波動度因為有群聚性，可預測性高出一個量級。',
        '此報告與主模型共用同一套特徵、walk-forward 切分與 embargo，只換標籤。',
        '',
        f'- 最佳模型：{best_name}',
        f'- 平均 AUC {best.auc.mean():.3f}、準確率 {best.accuracy.mean():.1%}、'
        f'基準（全猜多數類）{best.baseline.mean():.1%}',
        f'- 僅取模型最有把握的前 10%：準確率 {best.top_decile_accuracy.mean():.1%}',
        '',
        '## 各折',
        '',
        '| 期間 | AUC | 準確率 | 基準 | 正樣本率 | 前10%把握準確率 |',
        '|---|---|---|---|---|---|',
    ]
    for _, r in best.iterrows():
        lines.append(f'| {r.period} | {r.auc:.3f} | {r.accuracy:.1%} | {r.baseline:.1%} | '
                     f'{r.positive_rate:.1%} | {r.top_decile_accuracy:.1%} |')

    lines += ['', '## 各模型比較', '', '| 模型 | 平均 AUC | 準確率 | 基準 |', '|---|---|---|---|']
    for n, f in results.items():
        lines.append(f'| {n} | {f.auc.mean():.4f} | {f.accuracy.mean():.1%} | {f.baseline.mean():.1%} |')

    lines += [
        '',
        '> 準確率必須與同列的「基準」一起看。若準確率高只是因為正樣本率本來就高，那不是預測能力。',
        '> 僅為機器學習練習，不構成投資建議。',
        '',
    ]
    md = '\n'.join(lines)
    print()
    print(md)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / 'volatility.md').write_text(md, encoding='utf-8')
    best.to_csv(REPORT_DIR / 'volatility_folds.csv', index=False)
    print(f'已寫出 {REPORT_DIR / "volatility.md"}')


if __name__ == '__main__':
    main()
