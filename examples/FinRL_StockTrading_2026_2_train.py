"""
Stock NeurIPS2018 Part 2. Train (Phase 1 improved version)

本脚本是「DRL 训练效果改进计划」Phase 1 的核心实现。
和上一版相比，引入了 RL 工程上的几个标准最佳实践：

1. **划分验证集**：把 2014-01-06 ~ 2022-12-31 作训练集，2023-01-01 ~ 2024-01-01
   作验证集（held-out）。验证集只在 EvalCallback 里评估，不参与训练梯度。
2. **VecNormalize 包装**：对状态和奖励做 running mean/std 归一化，避免
   301 维状态向量里现金（1e6）、价格（1e2）、持仓数（1e2）量级悬殊导致 MLP 难收敛。
3. **EvalCallback + 早停**：每 5k 步在验证集上评估 episode reward，
   把验证集上表现最好的 checkpoint 保存为 best_model，避免训练过拟合到训练集。
4. **多种子训练**：对每个算法跑 3 个种子（42 / 123 / 2024），后续回测取均值±std，
   单种子结果在金融噪声数据上根本没有统计意义。
5. **turbulence 阈值自动校准**：用训练集 turbulence 序列的 99% 分位作阈值，
   而不是写死 70（之前 VIX 几乎从不超过 70，等同于没风控）。

学习重点：
- 理解 VecNormalize 为什么对 RL 几乎是必备的（参考 SB3 官方 RL Tips）
- 理解 EvalCallback 是怎么用验证集做模型选择的
- 理解多种子为什么是 RL 实验的"诚信底线"
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

from finrl.config import INDICATORS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

# Phase 3: 扩充因子
EXTRA_FACTORS = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d", "price_zscore_20", "volume_zscore_20", "sma_ratio_30_60",
]
ALL_INDICATORS = INDICATORS + EXTRA_FACTORS

# ============================================================
# 时间窗口（在脚本里就近定义，不动 finrl/config.py 全局常量，
#           以免影响 unit_tests 和其它 example 脚本）
# ============================================================
TRAIN_START = "2014-01-06"
TRAIN_END = "2023-01-01"  # data_split 是左闭右开，截止到 2022-12-30 的最后一个交易日
VAL_START = "2023-01-01"
VAL_END = "2024-01-01"

# 训练参数
SEEDS = [42, 123, 2024]
TOTAL_TIMESTEPS = 200_000  # Phase 2：100k 太短，提到 200k
EVAL_FREQ = 10_000          # 评估频率跟上

# 控制开关：Phase 4 只训练 RecurrentPPO（PPO/A2C 已在 Phase 3 训过，沿用）
ALGOS_TO_TRAIN = []
if HAS_RPPO:
    ALGOS_TO_TRAIN.append("rppo")


def set_global_seed(seed: int) -> None:
    """统一设置 random / numpy / torch 的种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(df: pd.DataFrame, env_kwargs: dict, turbulence_threshold: float | None):
    """构造一个 StockTradingEnv 并包装成 VecEnv。

    返回的是未经 VecNormalize 包装的 DummyVecEnv，
    由调用方决定如何 normalize（训练/验证共享同一份 obs_rms）。
    """
    e = StockTradingEnv(
        df=df,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col="vix",
        **env_kwargs,
    )
    return DummyVecEnv([lambda: e])


def main():
    # ---------- 0. 准备目录 ----------
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

    # ---------- 1. 加载并切分数据 ----------
    print(">>> Loading all_data_v2.csv …")
    df = pd.read_csv("all_data_v2.csv")
    df = df.set_index(df.columns[0])
    df.index.names = [""]

    train_df = data_split(df, TRAIN_START, TRAIN_END)
    val_df = data_split(df, VAL_START, VAL_END)
    print(
        f"train: {train_df['date'].min()} ~ {train_df['date'].max()}, "
        f"rows={len(train_df)}; val: {val_df['date'].min()} ~ "
        f"{val_df['date'].max()}, rows={len(val_df)}"
    )

    # ---------- 2. 计算环境配置 ----------
    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(ALL_INDICATORS) * stock_dimension
    print(f"stock_dim={stock_dimension}, state_space={state_space}, n_factors={len(ALL_INDICATORS)}")

    buy_cost_list = sell_cost_list = [0.0005] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 200,
        "initial_amount": 1_000_000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ALL_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,  # 仅在 reward_type='asset_diff' 时生效
        # Phase 2：使用 log return 奖励 + 少量 turnover 惩罚
        "reward_type": "log_return",
        "turnover_penalty": 0.001,
    }

    # ---------- 3. 自动校准 turbulence 阈值 ----------
    # 用训练集 turbulence 序列的 99% 分位做阈值。
    # 这样训练期间偶尔会触发清仓（约 1% 的极端日），让模型见过这种状态；
    # 同时不会像写死 70 那样从不触发，等同于没风控。
    train_turb_99 = float(train_df["turbulence"].quantile(0.99))
    train_vix_99 = float(train_df["vix"].quantile(0.99))
    print(
        f"turbulence 99% quantile = {train_turb_99:.2f}, "
        f"VIX 99% quantile = {train_vix_99:.2f}"
    )
    # 用 VIX（更稳健、和市场情绪挂钩），阈值取 99% 分位
    auto_threshold = train_vix_99

    # ---------- 4. 多算法 × 多种子训练循环 ----------
    for algo in ALGOS_TO_TRAIN:
        for seed in SEEDS:
            run_name = f"{algo}_seed{seed}"
            print(f"\n========== Training {run_name} ==========")
            set_global_seed(seed)

            # 4.1 构造训练环境 + VecNormalize
            train_vec = make_env(train_df, env_kwargs, auto_threshold)
            train_vec = VecNormalize(
                train_vec,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                gamma=0.99,
            )

            # 4.2 构造验证环境，共享 obs_rms（评估不更新统计量）
            val_vec_raw = make_env(val_df, env_kwargs, auto_threshold)
            val_vec = VecNormalize(
                val_vec_raw,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                training=False,
            )
            # EvalCallback 在每次评估前会自动 sync_envs_normalization，
            # 把 train_vec 的 obs_rms 拷贝到 val_vec。

            # 4.3 模型超参（基于 SB3 RL Zoo 经验，针对 301 维连续动作做小幅调整）
            if algo == "ppo":
                model_cls = PPO
                model_kwargs = dict(
                    n_steps=2048,
                    batch_size=128,
                    n_epochs=10,
                    learning_rate=3e-4,
                    ent_coef=0.005,
                    clip_range=0.2,
                    gae_lambda=0.95,
                    gamma=0.99,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[256, 256]),
                )
            elif algo == "a2c":
                model_cls = A2C
                model_kwargs = dict(
                    n_steps=64,            # 比默认 5 大很多，提高 advantage 估计质量
                    learning_rate=7e-4,
                    ent_coef=0.01,
                    gamma=0.99,
                    gae_lambda=0.95,
                    policy_kwargs=dict(net_arch=[256, 256]),
                )
            elif algo == "rppo":
                model_cls = RecurrentPPO
                model_kwargs = dict(
                    n_steps=512,            # LSTM 只要每次 rollout 竟足够长
                    batch_size=128,
                    n_epochs=10,
                    learning_rate=3e-4,
                    ent_coef=0.005,
                    clip_range=0.2,
                    gae_lambda=0.95,
                    gamma=0.99,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=[256],     # LSTM 后接 1 层 256
                        lstm_hidden_size=128,
                        n_lstm_layers=1,
                        enable_critic_lstm=True,
                    ),
                )
            else:
                raise ValueError(f"Unknown algo: {algo}")

            policy_name = "MlpLstmPolicy" if algo == "rppo" else "MlpPolicy"
            model = model_cls(
                policy=policy_name,
                env=train_vec,
                seed=seed,
                verbose=0,
                tensorboard_log=os.path.join(RESULTS_DIR, "tb"),
                **model_kwargs,
            )

            # 4.4 EvalCallback：每 EVAL_FREQ 步在验证集上评估，保留最优 checkpoint
            best_dir = os.path.join(TRAINED_MODEL_DIR, run_name)
            os.makedirs(best_dir, exist_ok=True)
            eval_callback = EvalCallback(
                val_vec,
                best_model_save_path=best_dir,
                log_path=best_dir,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=1,  # 一个 episode = 走完整个验证集
                deterministic=True,
                render=False,
                verbose=0,
            )

            # 4.5 训练
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                tb_log_name=run_name,
                progress_bar=False,
            )

            # 4.6 保存最终模型 + VecNormalize 统计量
            model.save(os.path.join(best_dir, "final_model.zip"))
            train_vec.save(os.path.join(best_dir, "vec_normalize.pkl"))
            print(
                f"saved: {best_dir}/best_model.zip (best on val) "
                f"+ final_model.zip + vec_normalize.pkl"
            )

    print("\n>>> All training done.")


if __name__ == "__main__":
    main()
