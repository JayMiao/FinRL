"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with baselines such as
Mean Variance Optimization and DJIA index.

学习重点：
1. 理解如何加载训练好的模型，在交易数据上做预测（回测）。
2. 理解 DRL_prediction 的内部流程：逐日喂数据 → 模型输出动作 → 环境执行买卖 → 计算账户价值。
3. 理解 turbulence_threshold 和 risk_indicator_col 的风控机制。
4. 理解 MVO 基准（均值方差优化）的计算原理，以及 DJIA 基准的作用。
5. 理解最终如何对比 5 个 DRL 模型 + 2 个基准 的表现。
"""

from __future__ import annotations

# Agg 后端用于非交互式画图（服务器无 GUI 时也能正常生成 PNG）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# stable-baselines3 提供的各算法模型类，用于 load 训练好的模型。
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

# DRLAgent 提供了 DRL_prediction 静态方法，用于回测预测。
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
# StockTradingEnv 用于构造回测环境，和训练使用的是同一个类。
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# %% Part 1. Load data

# 读取训练数据 CSV（MVO 基准需要用训练数据计算最优权重）。
train = pd.read_csv("train_data.csv")
# 读取交易数据 CSV（回测用的数据）。
trade = pd.read_csv("trade_data.csv")

# 处理索引：CSV 第一列是旧 index，设回索引。
train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents

# 控制哪些模型参与回测（必须和训练脚本中的开关保持一致，
# 否则对应的模型文件不存在会导致 load 报错）。
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

# 从 trained_models/ 目录加载训练好的模型。
# 每种算法对应一个 .zip 文件（stable-baselines3 的保存格式）。
# 如果对应开关为 False，则变量赋值为 None，后面跳过该模型。
trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

# %% Part 3. Backtesting - DRL agents

# --- 构造回测环境 ---
# 和训练环境相比，回测环境有两点关键不同：
#   1. 数据用 trade（交易集），而不是 train。
#   2. 加入了风控参数：turbulence_threshold 和 risk_indicator_col。

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

# 创建回测环境，注意两个风控参数（需要配合理解）：
#   turbulence_threshold=70：
#     这是一个通用风控阈值，对什么指标生效取决于 risk_indicator_col。
#     这里 risk_indicator_col="vix"，所以实际是 VIX >= 70 时触发清仓。
#     （注意：框架内部变量名叫 self.turbulence，但实际存的是 VIX 列的值，
#      这是 FinRL 命名不够严谨的地方，不要被变量名误导。）
#   risk_indicator_col="vix"：
#     指定用哪一列作为风控触发指标。默认是 "turbulence"，
#     这里改成 "vix"，即用波动率指数/恐慌指数作为风控依据。
#     当 VIX >= turbulence_threshold（70）时：
#       - 强制卖出所有持仓（清仓）
#       - 不允许买入
#     如果不传 turbulence_threshold，则不启用任何风控。
e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
)

# ============================================================
# DRL_prediction 工作流程（逐日推演）：
# 1. 通过 environment.get_sb_env() 获取 vectorized 环境。
# 2. 循环遍历每一天：
#    a. model.predict(test_obs) → 输出动作（对每只股票的买卖倾向）。
#    b. test_env.step(action) → 环境执行买卖，返回新状态和奖励。
# 3. 最后一步调用 save_asset_memory / save_action_memory 收集结果。
# 返回值：
#   df_account_value：每步的账户总资产。
#   df_actions：每步对每只股票的买卖动作。
# ============================================================

df_account_value_a2c, df_actions_a2c = (
    DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)
    if if_using_a2c
    else (None, None)
)

df_account_value_ddpg, df_actions_ddpg = (
    DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)
    if if_using_ddpg
    else (None, None)
)

df_account_value_ppo, df_actions_ppo = (
    DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if if_using_ppo
    else (None, None)
)

df_account_value_td3, df_actions_td3 = (
    DRLAgent.DRL_prediction(model=trained_td3, environment=e_trade_gym)
    if if_using_td3
    else (None, None)
)

df_account_value_sac, df_actions_sac = (
    DRLAgent.DRL_prediction(model=trained_sac, environment=e_trade_gym)
    if if_using_sac
    else (None, None)
)

# %% Part 4. Mean Variance Optimization baseline

# MVO（均值方差优化）是经典的组合优化方法。
# 核心思想：根据历史收益率和协方差矩阵，找到"给定风险下收益最大"
# 或"给定收益下风险最小"的资产权重分配。
# 这里作为 DRL 模型的对比基准。


def process_df_for_mvo(df):
    """将 FinRL 的长格式 DataFrame 转为 MVO 需要的宽格式。
    长格式：每行是一个 (日期, 股票) 组合。
    宽格式：每行是一个日期，每列是一只股票的收盘价。
    例如：
      长格式: date=T, tic=AAPL, close=150
               date=T, tic=MSFT, close=300
      宽格式: index=T, AAPL=150, MSFT=300
    """
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    """手动计算每只股票的日收益率矩阵。
    收益率 = (当日价格 - 前一日价格) / 前一日价格 × 100。
    返回值形状为 (Rows-1, Columns)，因为第一天没有收益率。
    """
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100
    return StockReturn


# 将训练数据转为宽格式（每列一只股票的价格时间序列）。
StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

# 计算训练期每只股票的日收益率。
arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

# 计算平均收益率向量和协方差矩阵。
# meanReturns：每只股票的历史平均日收益率。
meanReturns = np.mean(arReturns, axis=0)
# covReturns：股票之间的协方差矩阵，衡量股票间的联动关系。
# rowvar=False 表示每列是一个变量（股票），不是每行。
covReturns = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

from pypfopt.efficient_frontier import EfficientFrontier

# 用 PyPortfolioOpt 库构造有效前沿。
# weight_bounds=(0, 0.5)：每只股票权重在 0 到 50% 之间（防止过度集中）。
ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
# max_sharpe()：在有效前沿上找到夏普比率最大的点。
# 夏普比率 = (收益 - 无风险利率) / 波动率，衡量单位风险的回报。
raw_weights_mean = ef_mean.max_sharpe()
# clean_weights()：把小到可以忽略的权重清零，使结果更简洁。
cleaned_weights_mean = ef_mean.clean_weights()
# 把权重比例转换为实际投资金额（总资金 100 万 × 权重）。
mvo_weights = np.array(
    [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
)

# 计算初始持仓股数：投资金额 / 训练期最后一天的股价。
# 1/p 就是"用 1 美元能买多少股"，再乘以投资金额得到实际股数。
LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

# 在交易数据上模拟：每天的持仓股数 × 每天的股价 = 每天的资产价值。
Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

# %% Part 5. DJIA index baseline

# 下载道琼斯工业平均指数（^DJI）作为第二个基准。
# 这代表"买指数并持有"的被动策略。
import yfinance as yf

df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
# 只保留收盘价，重置索引让日期变成列。
df_dji = df_dji[["Close"]].reset_index()
df_dji.columns = ["date", "close"]
df_dji["date"] = df_dji["date"].astype(str)
# 归一化处理：以第一天为基准（100 万），计算每天的相对价值。
# 这样 DJIA 基准和 DRL 模型的初始资金都是 100 万，可以直接对比。
fst_day = df_dji["close"].iloc[0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

# %% Part 6. Compare results

# 把各模型的结果 DataFrame 设为以日期为索引，方便对齐合并。
df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c
    else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg
    else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo
    else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3
    else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac
    else None
)

# 把所有结果合并到一张表里，每列是一个策略的账户价值时间序列。
# 列说明：
#   a2c/ddpg/ppo/td3/sac → 5 种 DRL 算法的回测结果。
#   mvo → 均值方差优化的组合。
#   dji → 道琼斯指数的买入持有。
result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
        "mvo": MVO_result["Mean Var"],
        "dji": dji["close"],
    }
)

print("\n=== Backtest Results ===")
print(result)

# %% Part 7. Plot

# 设置图形大小（宽 15 英寸、高 5 英寸）。
plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
# 画出所有策略的资产曲线对比图。
# 横轴是日期，纵轴是资产价值。
# 哪条线在最上面，说明该策略在回测期间表现最好。
result.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
# 保存为 PNG 图片。
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to backtest_result.png")
