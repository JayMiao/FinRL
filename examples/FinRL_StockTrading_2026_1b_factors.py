"""
Phase 3：在 all_data.csv 之上扩充因子，生成 all_data_v2.csv。

新增因子（共 7 个），均按 tic 分组安全计算，生成在 [date, tic] 面板里：
  - ret_1d / ret_5d / ret_20d   动量族
  - vol_20d                       20 日波动率
  - price_zscore_20               收盘价 20 日 z-score（均值回归代理）
  - volume_zscore_20              成交量 20 日 z-score
  - sma_ratio_30_60               30 日 SMA / 60 日 SMA（短中趋势对比）

策略意图：给 DRL 一些它原本看不到的「截面 / 时序结构」信号，
特别是 ret_5d / ret_20d 是经典的 cross-sectional momentum，
能帮模型在牛/震荡市做好选股。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SRC = "all_data.csv"
DST = "all_data_v2.csv"

# 新增因子列名（顺序固定，后续脚本会把它们追加到 INDICATORS 之后）
NEW_FACTORS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "price_zscore_20",
    "volume_zscore_20",
    "sma_ratio_30_60",
]


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """按 tic 分组安全地添加新因子。"""
    df = df.copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    g = df.groupby("tic", group_keys=False)

    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_20d"] = g["close"].pct_change(20)

    df["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std())

    def zscore(s: pd.Series, w: int) -> pd.Series:
        m = s.rolling(w, min_periods=5).mean()
        sd = s.rolling(w, min_periods=5).std()
        return (s - m) / (sd.replace(0, np.nan))

    df["price_zscore_20"] = g["close"].transform(lambda s: zscore(s, 20))
    df["volume_zscore_20"] = g["volume"].transform(lambda s: zscore(s, 20))

    # 30/60 SMA 比值。close_30_sma / close_60_sma 已存在 → 直接相除即可
    df["sma_ratio_30_60"] = (df["close_30_sma"] / df["close_60_sma"]).replace(
        [np.inf, -np.inf], np.nan
    )

    # 缺失值处理：按 tic 前向填充，再统一填 0（仅前几行受影响）
    for col in NEW_FACTORS:
        df[col] = g[col].transform(lambda s: s.ffill())
    df[NEW_FACTORS] = df[NEW_FACTORS].fillna(0)

    # 保持原排序（date, tic）便于环境消费
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    return df


def main():
    print(f">>> Loading {SRC} …")
    df = pd.read_csv(SRC)
    if df.columns[0] in ("Unnamed: 0", ""):
        df = df.drop(columns=df.columns[0])

    print(f"input rows={len(df)}, cols={len(df.columns)}")
    df2 = add_factors(df)
    print(
        f"output rows={len(df2)}, cols={len(df2.columns)}; new factors: "
        + ", ".join(NEW_FACTORS)
    )
    print(df2[["date", "tic"] + NEW_FACTORS].tail(8).to_string(index=False))

    df2.to_csv(DST, index=True)
    print(f">>> Saved to {DST}")


if __name__ == "__main__":
    main()
