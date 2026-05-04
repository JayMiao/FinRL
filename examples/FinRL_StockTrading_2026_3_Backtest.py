"""
Stock NeurIPS2018 Part 3. Backtest (Phase 1 multi-window evaluation)

本脚本配合改进版的 train 脚本使用，完成 Phase 1 的「评测协议」环节：
- 在 3 个测试窗口（牛/震/熊）上评估每个 (algo, seed) 组合
- 输出多种子均值 ± 标准差的对比表
- 同时和 MVO + DJIA 基线对比

测试窗口设计依据：
- 2024-Q1（牛市段）：科技股推动美股上涨，DJIA 普遍走强
- 2024-Q4（震荡段）：年底窄幅震荡，趋势策略容易失效
- 2026-Q1（熊市段）：当前现状窗口，DJIA -4.9%

学习重点：
1. 理解为什么单窗口测试不够：金融数据噪声大、市况切换剧烈，
   单一窗口结论很可能是「运气」。
2. 理解多种子的意义：RL 训练对随机种子极敏感，单种子结果不能代表算法能力。
3. 理解三大评估指标：
   - 累计收益（直观但不抗风险）
   - 年化 Sharpe（单位风险回报，> 1 算可用，> 1.5 算优秀）
   - 最大回撤 MDD（风险维度，越接近 0 越好）
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

EXTRA_FACTORS = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d", "price_zscore_20", "volume_zscore_20", "sma_ratio_30_60",
]
ALL_INDICATORS = INDICATORS + EXTRA_FACTORS

# ============================================================
# 评测窗口与种子配置（必须和 2_train.py 中保持一致）
# ============================================================
TEST_WINDOWS = [
    ("2024Q1_bull", "2024-01-01", "2024-04-01"),
    ("2024Q4_chop", "2024-10-01", "2025-01-01"),
    ("2026Q1_bear", "2026-01-01", "2026-03-20"),
]
SEEDS = [42, 123, 2024]
ALGOS = ["ppo", "a2c"]
if HAS_RPPO:
    ALGOS.append("rppo")

TRAIN_START = "2014-01-06"
TRAIN_END = "2023-01-01"

INITIAL_CAPITAL = 1_000_000


# ============================================================
# 工具函数：指标计算
# ============================================================
def compute_metrics(asset_series: pd.Series) -> dict:
    """根据账户资产时间序列计算评估指标。

    输入 asset_series：index 是日期，值是当日总资产。
    返回 dict 含：cum_return / annual_return / sharpe / max_drawdown
    """
    asset_series = asset_series.dropna()
    if len(asset_series) < 2:
        return {"cum_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    daily_ret = asset_series.pct_change().dropna()
    cum_return = asset_series.iloc[-1] / asset_series.iloc[0] - 1.0

    # 年化（按 252 交易日）
    n = len(daily_ret)
    annual_return = (1 + cum_return) ** (252.0 / n) - 1.0
    annual_vol = daily_ret.std() * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol > 1e-9 else np.nan

    # 最大回撤
    cum_max = asset_series.cummax()
    drawdown = asset_series / cum_max - 1.0
    max_dd = drawdown.min()

    return {
        "cum_return": float(cum_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


# ============================================================
# 工具函数：构造测试环境（不带 VecNormalize 包装，由调用方加载）
# ============================================================
def build_test_env(test_df: pd.DataFrame, env_kwargs: dict, turbulence_threshold: float):
    e = StockTradingEnv(
        df=test_df,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col="vix",
        **env_kwargs,
    )
    return e, DummyVecEnv([lambda: e])


# ============================================================
# 工具函数：手动预测循环（兼容 VecNormalize 加载的统计量）
# ============================================================
def run_drl_prediction(model, raw_env: StockTradingEnv, vec_normalize_path: str,
                         is_recurrent: bool = False):
    """加载训练时的 obs running stats，手动跑完整测试集。

    注意：直接用 DummyVecEnv 包装会在 terminal 触发自动 reset，把 raw_env
    的 asset_memory 清空。这里我们手动读取 VecNormalize 的 obs_rms，自己归一化。
    """
    tmp = DummyVecEnv([lambda: raw_env])
    vn = VecNormalize.load(vec_normalize_path, tmp)
    obs_rms = vn.obs_rms
    clip_obs = vn.clip_obs
    epsilon = vn.epsilon

    def normalize(o: np.ndarray) -> np.ndarray:
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                       -clip_obs, clip_obs).astype(np.float32)

    obs = raw_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.asarray(obs, dtype=np.float32)

    # RecurrentPPO 需要隐藏状态
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    done = False
    while not done:
        norm_obs = normalize(obs).reshape(1, -1)
        if is_recurrent:
            action, lstm_states = model.predict(
                norm_obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(norm_obs, deterministic=True)
        step_out = raw_env.step(action[0])
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, _, done, _ = step_out
            done = bool(done)
        obs = np.asarray(obs, dtype=np.float32)

    asset_memory = raw_env.asset_memory
    date_memory = raw_env.date_memory
    n = min(len(asset_memory), len(date_memory))
    s = pd.Series(asset_memory[:n], index=pd.to_datetime(date_memory[:n]))
    return s


# ============================================================
# MVO 基线
# ============================================================
def compute_mvo(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    from pypfopt.efficient_frontier import EfficientFrontier

    train_wide = train_df.pivot(index="date", columns="tic", values="close")
    test_wide = test_df.pivot(index="date", columns="tic", values="close")

    arr = np.asarray(train_wide)
    rows, cols = arr.shape
    rets = np.zeros((rows - 1, cols))
    for j in range(cols):
        for i in range(rows - 1):
            rets[i, j] = (arr[i + 1, j] - arr[i, j]) / arr[i, j] * 100.0

    mean_ret = pd.Series(np.mean(rets, axis=0), index=train_wide.columns)
    cov_ret = pd.DataFrame(np.cov(rets, rowvar=False),
                            index=train_wide.columns, columns=train_wide.columns)

    ef = EfficientFrontier(mean_ret, cov_ret, weight_bounds=(0, 0.5))
    ef.max_sharpe()
    weights = ef.clean_weights()
    w_arr = np.array([INITIAL_CAPITAL * weights[c] for c in train_wide.columns])
    last_price = train_wide.iloc[-1].to_numpy()
    init_shares = w_arr / last_price

    portfolio = test_wide @ init_shares
    portfolio.index = pd.to_datetime(portfolio.index)
    return portfolio


# ============================================================
# DJIA 基线
# ============================================================
def compute_djia(start: str, end: str) -> pd.Series:
    df = yf.download("^DJI", start=start, end=end, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    df = df[["Close"]].reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    fst = float(df["close"].iloc[0])
    s = df.set_index("date")["close"] / fst * INITIAL_CAPITAL
    s.name = "dji"
    return s


# ============================================================
# 主流程
# ============================================================
def main():
    print(">>> Loading all_data_v2.csv …")
    df = pd.read_csv("all_data_v2.csv")
    df = df.set_index(df.columns[0])
    df.index.names = [""]

    # 用训练集 VIX 99% 分位作 threshold（和训练保持一致）
    train_df = data_split(df, TRAIN_START, TRAIN_END)
    threshold = float(train_df["vix"].quantile(0.99))
    print(f"VIX 99% threshold = {threshold:.2f}")

    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(ALL_INDICATORS) * stock_dimension
    env_kwargs = {
        "hmax": 200,
        "initial_amount": INITIAL_CAPITAL,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.0005] * stock_dimension,
        "sell_cost_pct": [0.0005] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ALL_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "reward_type": "log_return",
        "turnover_penalty": 0.001,
    }

    summary_rows = []  # 最终汇总表

    for win_name, win_start, win_end in TEST_WINDOWS:
        print(f"\n========== Window {win_name}: {win_start} ~ {win_end} ==========")
        test_df = data_split(df, win_start, win_end)
        if len(test_df) < 30:
            print(f"  skipped (too few rows: {len(test_df)})")
            continue

        # --- 各 DRL 模型 ---
        all_curves = {}  # column name -> series
        for algo in ALGOS:
            for seed in SEEDS:
                run = f"{algo}_seed{seed}"
                model_path = os.path.join(TRAINED_MODEL_DIR, run, "best_model.zip")
                vn_path = os.path.join(TRAINED_MODEL_DIR, run, "vec_normalize.pkl")
                if not os.path.exists(model_path) or not os.path.exists(vn_path):
                    print(f"  [SKIP] {run}: model or vec_normalize not found")
                    continue
                if algo == "ppo":
                    cls = PPO; is_rec = False
                elif algo == "a2c":
                    cls = A2C; is_rec = False
                else:  # rppo
                    cls = RecurrentPPO; is_rec = True
                model = cls.load(model_path, device="cpu")
                raw_env, _ = build_test_env(test_df, env_kwargs, threshold)
                series = run_drl_prediction(model, raw_env, vn_path, is_recurrent=is_rec)
                metrics = compute_metrics(series)
                summary_rows.append({
                    "window": win_name, "strategy": algo, "seed": seed, **metrics
                })
                all_curves[f"{algo}_s{seed}"] = series
                print(
                    f"  {run}: cum={metrics['cum_return']*100:+.2f}% "
                    f"sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']*100:.2f}%"
                )

        # --- Phase 5 集成：所有 algo×seed 资产曲线等权平均（按资产重新归一化后算复合收益） ---
        drl_curves = [s for k, s in all_curves.items()
                      if any(k.startswith(a + "_") for a in ALGOS)]
        if len(drl_curves) >= 2:
            ens_df = pd.concat([s / s.iloc[0] for s in drl_curves], axis=1).dropna()
            ens_series = ens_df.mean(axis=1) * INITIAL_CAPITAL
            ens_metrics = compute_metrics(ens_series)
            summary_rows.append({
                "window": win_name, "strategy": "ensemble", "seed": "-", **ens_metrics
            })
            all_curves["ensemble"] = ens_series
            print(
                f"  ensemble: cum={ens_metrics['cum_return']*100:+.2f}% "
                f"sharpe={ens_metrics['sharpe']:.2f} mdd={ens_metrics['max_drawdown']*100:.2f}%"
            )

        # --- MVO 基线 ---
        mvo = compute_mvo(train_df, test_df)
        m_metrics = compute_metrics(mvo)
        summary_rows.append({
            "window": win_name, "strategy": "mvo", "seed": "-", **m_metrics
        })
        all_curves["mvo"] = mvo
        print(f"  mvo: cum={m_metrics['cum_return']*100:+.2f}% "
              f"sharpe={m_metrics['sharpe']:.2f} mdd={m_metrics['max_drawdown']*100:.2f}%")

        # --- DJIA 基线 ---
        dji = compute_djia(win_start, win_end)
        if len(dji) > 0:
            d_metrics = compute_metrics(dji)
            summary_rows.append({
                "window": win_name, "strategy": "dji", "seed": "-", **d_metrics
            })
            all_curves["dji"] = dji
            print(f"  dji: cum={d_metrics['cum_return']*100:+.2f}% "
                  f"sharpe={d_metrics['sharpe']:.2f} mdd={d_metrics['max_drawdown']*100:.2f}%")

        # --- 画图 ---
        fig, ax = plt.subplots(figsize=(15, 5))
        for name, s in all_curves.items():
            ax.plot(s.index, s.values, label=name, alpha=0.6 if name.startswith(("ppo_", "a2c_")) else 1.0)
        ax.set_title(f"Portfolio Value: {win_name}")
        ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"backtest_{win_name}.png", dpi=120)
        plt.close(fig)

    # ============================================================
    # 汇总表：按 (window, strategy) 取多种子均值±std
    # ============================================================
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("backtest_summary_raw.csv", index=False)
    print("\n>>> Raw per-seed table saved to backtest_summary_raw.csv")

    # 聚合多种子
    agg_rows = []
    for (win, strat), sub in summary.groupby(["window", "strategy"]):
        if len(sub) > 1:
            agg_rows.append({
                "window": win,
                "strategy": strat,
                "n_seeds": len(sub),
                "cum_return_mean": sub["cum_return"].mean(),
                "cum_return_std": sub["cum_return"].std(),
                "sharpe_mean": sub["sharpe"].mean(),
                "sharpe_std": sub["sharpe"].std(),
                "mdd_mean": sub["max_drawdown"].mean(),
            })
        else:
            agg_rows.append({
                "window": win,
                "strategy": strat,
                "n_seeds": 1,
                "cum_return_mean": sub["cum_return"].iloc[0],
                "cum_return_std": 0.0,
                "sharpe_mean": sub["sharpe"].iloc[0],
                "sharpe_std": 0.0,
                "mdd_mean": sub["max_drawdown"].iloc[0],
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv("backtest_summary_agg.csv", index=False)

    print("\n========== AGGREGATE SUMMARY (mean across seeds) ==========")
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(agg.to_string(index=False))

    # PASS/FAIL: DRL 平均累计收益 ≥ MVO？（在 sharpe 上比较更稳）
    print("\n========== Phase 1 PASS/FAIL ==========")
    for win in agg["window"].unique():
        w = agg[agg["window"] == win]
        mvo_sh = w[w["strategy"] == "mvo"]["sharpe_mean"].values
        for strat in ALGOS:
            row = w[w["strategy"] == strat]
            if len(row) and len(mvo_sh):
                drl_sh = row["sharpe_mean"].values[0]
                ok = "PASS" if drl_sh >= mvo_sh[0] else "FAIL"
                print(f"  {win} | {strat}: sharpe {drl_sh:+.2f} vs mvo {mvo_sh[0]:+.2f}  [{ok}]")


if __name__ == "__main__":
    main()
