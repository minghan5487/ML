import numpy as np
import pandas as pd

from trend.config import HORIZON, MARKET, STOCKS
from trend.data import load, load_chip, load_margin, load_revenue, load_valuation

TECHNICAL = [
    'ret_5', 'ret_20', 'ret_60', 'ma_gap_20', 'ma_gap_60', 'rsi_14', 'macd_hist',
    'volatility_20', 'volume_ratio', 'hl_range', 'rel_strength_20',
]
EXPERT = [
    'faber_trend_200', 'mom_12_1',
    'per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y',
    'breadth_200',
    'mkt_drawdown_250', 'mkt_trend_200',
    'high_52w_gap', 'vwap_gap_20',
]
# 非價格資料：籌碼面與基本面。全部以「當日已可取得」為原則對齊，
# 月營收另加公布落後天數（見 config.REVENUE_PUBLISH_LAG_DAYS）。
FLOW = [
    'foreign_net_5', 'foreign_net_20', 'trust_net_5', 'trust_net_20',
    'margin_chg_5', 'margin_chg_20', 'short_margin_ratio',
    'revenue_yoy', 'revenue_yoy_chg',
]
FEATURES = TECHNICAL + EXPERT + FLOW
# 這些欄位允許缺值（由 imputer 處理），不納入 dropna 條件
OPTIONAL = ['per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y'] + FLOW
REQUIRED = [f for f in FEATURES if f not in OPTIONAL]


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def price_features(df):
    close, volume = df['Close'], df['Volume']
    ret = close.pct_change()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    vwap_20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum()

    return pd.DataFrame({
        'ret_5': close.pct_change(5),
        'ret_20': close.pct_change(20),
        'ret_60': close.pct_change(60),
        'ma_gap_20': close / close.rolling(20).mean() - 1,
        'ma_gap_60': close / close.rolling(60).mean() - 1,
        'rsi_14': rsi(close),
        'macd_hist': (macd - macd.ewm(span=9, adjust=False).mean()) / close,
        'volatility_20': ret.rolling(20).std(),
        'volume_ratio': volume / volume.rolling(20).mean(),
        'hl_range': ((df['High'] - df['Low']) / close).rolling(20).mean(),
        'faber_trend_200': close / close.rolling(200).mean() - 1,
        'mom_12_1': close.shift(21) / close.shift(252) - 1,
        'high_52w_gap': close / close.rolling(252).max() - 1,
        'vwap_gap_20': close / vwap_20 - 1,
    }, index=df.index)


def valuation_features(code, index):
    val = load_valuation(code)
    if val is None or val.empty:
        return pd.DataFrame(index=index, columns=['per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y'], dtype=float)

    val = val.reindex(index).ffill()
    per = val['PER'].where(val['PER'] > 0)
    pbr = val['PBR'].where(val['PBR'] > 0)
    return pd.DataFrame({
        'per': per,
        'pbr': pbr,
        'dividend_yield': val['dividend_yield'],
        'per_vs_3y': per / per.rolling(750, min_periods=250).median() - 1,
        'pbr_vs_3y': pbr / pbr.rolling(750, min_periods=250).median() - 1,
    }, index=index)


MARKET_FFILL_LIMIT = 3


def flow_features(code, index, volume):
    """籌碼與基本面特徵。index 為該股交易日，volume 用來把法人買賣超正規化。"""
    out = pd.DataFrame(index=index)

    chip = load_chip(code)
    if chip is not None and not chip.empty:
        chip = chip.reindex(index)
        vol20 = volume.rolling(20).mean()
        for side in ('foreign', 'trust'):
            col = f'{side}_net'
            if col in chip.columns:
                net = chip[col].fillna(0)
                out[f'{side}_net_5'] = net.rolling(5).sum() / vol20
                out[f'{side}_net_20'] = net.rolling(20).sum() / vol20

    margin = load_margin(code)
    if margin is not None and not margin.empty:
        margin = margin.reindex(index).ffill(limit=5)
        bal = margin['margin_balance']
        out['margin_chg_5'] = bal / bal.shift(5) - 1
        out['margin_chg_20'] = bal / bal.shift(20) - 1
        out['short_margin_ratio'] = margin['short_balance'] / bal.replace(0, np.nan)

    rev = load_revenue(code)
    if rev is not None and not rev.empty and 'available_at' in rev.columns:
        rev = rev.copy()
        rev['available_at'] = pd.to_datetime(rev['available_at'])
        rev['yoy'] = rev['revenue'] / rev['revenue'].shift(12) - 1
        # 以「最早可用日」為索引往後填補，確保任一交易日只看得到當時已公布的營收
        avail = rev.dropna(subset=['yoy']).set_index('available_at')[['yoy']].sort_index()
        avail = avail[~avail.index.duplicated(keep='last')]
        aligned = avail.reindex(avail.index.union(index)).ffill().reindex(index)
        out['revenue_yoy'] = aligned['yoy']
        out['revenue_yoy_chg'] = aligned['yoy'] - aligned['yoy'].shift(20)

    for col in FLOW:
        if col not in out.columns:
            out[col] = np.nan
    return out[FLOW]


def market_features():
    close = load(MARKET)['Close']
    return pd.DataFrame({
        'mkt_ret_20': close.pct_change(20),
        'mkt_drawdown_250': close / close.rolling(250).max() - 1,
        'mkt_trend_200': close / close.rolling(200).mean() - 1,
    })


def build_dataset():
    market = market_features()
    frames = {}

    for code in STOCKS:
        df = load(f'{code}.TW')
        if df is None or len(df) < 300:
            continue
        # 大盤資料可能比個股晚一個交易日；向前填補至多 MARKET_FFILL_LIMIT 天，
        # 否則當日所有個股會因 mkt_* 為 NaN 而被 dropna 整批刪除（預測永遠慢一天）。
        feats = price_features(df).join(market.reindex(df.index).ffill(limit=MARKET_FFILL_LIMIT), how='left')
        feats = feats.join(valuation_features(code, df.index))
        feats = feats.join(flow_features(code, df.index, df['Volume']))
        feats['rel_strength_20'] = feats['ret_20'] - feats['mkt_ret_20']
        feats['future_return'] = df['Close'].shift(-HORIZON) / df['Close'] - 1
        # 波動度標籤：未來 HORIZON 日實現波動 vs 過去 HORIZON 日
        ret = df['Close'].pct_change()
        # ret.shift(-H) 在 t 取到 ret[t+H]；再 rolling(H) 即 ret[t+1..t+H] 的標準差
        feats['future_vol'] = ret.shift(-HORIZON).rolling(HORIZON).std()
        feats['past_vol'] = ret.rolling(HORIZON).std()
        feats['close'] = df['Close']
        feats['code'] = code
        frames[code] = feats

    above_200 = pd.concat({c: f['faber_trend_200'] > 0 for c, f in frames.items()}, axis=1)
    valid = pd.concat({c: f['faber_trend_200'].notna() for c, f in frames.items()}, axis=1)
    breadth = above_200.sum(axis=1).astype(float) / valid.sum(axis=1).replace(0, np.nan)

    data = pd.concat(frames.values())
    data['breadth_200'] = breadth.reindex(data.index).values
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    data.index.name = 'date'
    data['label'] = (data['future_return'] > 0).astype(int)
    # 波動度放大與否（未來 20 日實現波動 > 過去 20 日），用於另一條可預測性較高的標的
    data['label_vol'] = (data['future_vol'] > data['past_vol']).astype(int)

    labeled = data[data['future_return'].notna()].sort_index()
    latest = data[data.index == data.index.max()]
    return labeled, latest


def expert_views(row):
    views = {}

    views['Meb Faber'] = ('多', '股價在 200 日均線之上') if row.faber_trend_200 > 0 else ('空', '股價跌破 200 日均線')

    if pd.isna(row.per_vs_3y):
        views['Aswath Damodaran'] = ('—', '無估值資料')
    elif row.per_vs_3y < -0.15:
        views['Aswath Damodaran'] = ('多', f'本益比低於自身 3 年中位數 {abs(row.per_vs_3y):.0%}')
    elif row.per_vs_3y > 0.3:
        views['Aswath Damodaran'] = ('空', f'本益比高於自身 3 年中位數 {row.per_vs_3y:.0%}')
    else:
        views['Aswath Damodaran'] = ('中性', '本益比接近歷史水準')

    if row.breadth_200 > 0.6:
        views['Charlie Bilello'] = ('多', f'{row.breadth_200:.0%} 權值股站上 200 日線，市場廣度健康')
    elif row.breadth_200 < 0.4:
        views['Charlie Bilello'] = ('空', f'僅 {row.breadth_200:.0%} 權值股站上 200 日線')
    else:
        views['Charlie Bilello'] = ('中性', f'{row.breadth_200:.0%} 權值股站上 200 日線')

    if row.mkt_drawdown_250 < -0.15:
        views['Morgan Housel'] = ('多', f'大盤自高點回檔 {abs(row.mkt_drawdown_250):.0%}，長期投資者別恐慌')
    else:
        views['Morgan Housel'] = ('中性', '市場未明顯恐慌，維持長期紀律')

    if row.high_52w_gap > -0.03 and row.vwap_gap_20 > 0:
        views['Peter Brandt'] = ('多', '接近 52 週新高且站上 20 日 VWAP')
    elif row.high_52w_gap < -0.2 and row.vwap_gap_20 < 0:
        views['Peter Brandt'] = ('空', f'距 52 週高點 {abs(row.high_52w_gap):.0%} 且低於 VWAP')
    else:
        views['Peter Brandt'] = ('中性', '未出現明確突破')

    return views
