"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.

学习重点：
1. 先理解 FinRL 是如何把原始行情下载成统一格式的。
2. 再理解它如何补充技术指标、VIX、turbulence 等特征。
3. 最后理解如何把完整数据切分成训练集和交易集，供后续训练脚本直接使用。
"""

from __future__ import annotations

# itertools.product 用来生成日期和股票代码的笛卡尔积，
# 后面会用它补齐“每个交易日 x 每只股票”的完整面板数据。
import itertools

# pandas 是这个脚本里最核心的数据处理库，DataFrame 基本都靠它操作。
import pandas as pd
# yfinance 是直接从 Yahoo Finance 下载数据的第三方库。
# 这里先直接调用一次，方便你和 FinRL 自己封装的下载器做对比。
import yfinance as yf

# config_tickers 中保存了预定义股票池，这里会使用道琼斯 30 成分股列表。
from finrl import config_tickers
# INDICATORS 是 FinRL 预设的一组技术指标名称，例如 MACD、RSI、CCI、ADX 等。
from finrl.config import INDICATORS
# TRADE_END_DATE / TRADE_START_DATE 定义交易（回测）区间。
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
# TRAIN_END_DATE / TRAIN_START_DATE 定义模型训练区间。
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
# data_split 是 FinRL 的数据切分工具，会按日期范围筛选并重新整理索引。
from finrl.meta.preprocessor.preprocessors import data_split
# FeatureEngineer 是 FinRL 的特征工程入口，负责技术指标、VIX、turbulence 等扩展特征。
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
# YahooDownloader 是 FinRL 对 Yahoo Finance 下载流程的封装，
# 会把列名整理成框架统一使用的格式。
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# %% Part 1. Fetch data - Single ticker

# 第 1 部分：先只下载一只股票，观察“原生 yfinance”与“FinRL 封装下载器”的差异。

# 使用 yfinance 直接下载苹果（AAPL）在 2020-01-01 到 2020-01-31 之间的数据。
# 参数说明：
# - tickers="aapl"：要下载的股票代码。
# - start="2020-01-01"：起始日期，包含这一天。
# - end="2020-01-31"：结束日期，通常按 yfinance 习惯是右开区间，
#   所以实际最后一天常常不会包含 2020-01-31 当天。
aapl_df_yf = yf.download(tickers="aapl", start="2020-01-01", end="2020-01-31")
# 打印分隔标题，便于在终端里区分不同阶段的输出。
print("=== yfinance download ===")
# head() 默认显示前 5 行，适合快速看数据结构。
print(aapl_df_yf.head())

# 使用 FinRL 的 YahooDownloader 下载同样的数据。
# 和直接使用 yfinance 相比，它的价值主要在于：
# 1. 统一列名格式，方便后续环境与训练代码直接消费。
# 2. 自动补充 tic、day 等字段。
# 3. 在需要时对价格做 adjusted close 对齐处理。
aapl_df_finrl = YahooDownloader(
    # 下载起始日期。
    start_date="2020-01-01",
    # 下载结束日期。
    end_date="2020-01-31",
    # ticker_list 必须是列表，即使只下载一只股票也写成 ["aapl"]。
    ticker_list=["aapl"],
# fetch_data() 真正发起下载请求并返回整理后的 DataFrame。
).fetch_data()
print("\n=== FinRL YahooDownloader ===")
print(aapl_df_finrl.head())

# %% Part 2. Fetch data - DOW 30 tickers

# 第 2 部分：把单股票扩展到多股票。
# FinRL 的股票交易例子通常使用道琼斯 30 成分股作为示例股票池。
print("\n=== DOW 30 Tickers ===")
# 直接打印股票池，先知道后面会下载哪些股票。
print(config_tickers.DOW_30_TICKER)

# 下载从训练开始日期到交易结束日期的完整原始行情数据。
# 这样做的目的是一次性拿到“训练 + 交易/回测”两个阶段所需的全部历史数据。
df_raw = YahooDownloader(
    # 训练区间起点。
    start_date=TRAIN_START_DATE,
    # 交易区间终点。
    end_date=TRADE_END_DATE,
    # 股票列表使用 FinRL 预定义的 DOW 30 成分股。
    ticker_list=config_tickers.DOW_30_TICKER,
).fetch_data()
print("\n=== Raw data ===")
# 这里看到的是“原始但已标准化”的行情数据，
# 一般已经包含 date/open/high/low/close/volume/tic/day 等列。
print(df_raw.head())

# %% Part 3. Preprocess data

# 第 3 部分：特征工程。
# 这是 FinRL 数据流程里最关键的一步，因为强化学习环境不只需要 OHLCV，
# 往往还要配合技术指标和市场状态特征一起作为状态空间输入。
fe = FeatureEngineer(
    # True 表示计算技术指标。
    use_technical_indicator=True,
    # 要计算哪些技术指标，直接复用 FinRL 默认配置。
    tech_indicator_list=INDICATORS,
    # True 表示加入 VIX（波动率指数）作为市场恐慌程度代理变量。
    use_vix=True,
    # True 表示加入 turbulence 指标，用于描述市场异常波动程度。
    use_turbulence=True,
    # user_defined_feature：是否启用自定义因子。
    # 如果设为 True，preprocess_data() 内部会自动调用 add_user_defined_feature()。
    #
    # ============================================================
    # 【扩展指南】如何通过 FeatureEngineer 加入自己挖掘的因子：
    # ============================================================
    # 第 1 步：打开 finrl/meta/preprocessor/preprocessors.py，
    #         找到 add_user_defined_feature(self, data) 方法。
    #         在 return df 之前添加你的因子计算逻辑。
    #
    #         例如加一个 "5 日价格动量因子"：
    #         df["momentum_5"] = df.groupby("tic")["close"].pct_change(5)
    #
    #         ⚠️ 注意：数据中有多只股票，所以计算时必须先 groupby("tic")
    #         再操作，否则 shift/pct_change 会跨股票串行，算出错误值。
    #
    #         你可以添加任意多个因子，只要最终 df 中多了对应列即可：
    #         df["my_factor_1"] = ...
    #         df["my_factor_2"] = ...
    #
    # 第 2 步：把下面的 False 改成 True，让 FeatureEngineer 在
    #         preprocess_data() 流程中自动调用你写的 add_user_defined_feature()。
    #         生成的 processed 中就会包含你新增的因子列。
    #
    # 第 3 步（重要）：加了新因子后，训练环境的状态空间维度也要同步更新。
    #         训练脚本中 state_space 的计算公式需要加上你的因子列数：
    #         state_space = 1 + 2*stock_dimension + (len(INDICATORS)+你增加的因子数)*stock_dimension
    #         否则环境初始化时会因维度不匹配而报错。
    # ============================================================
    user_defined_feature=False,
)

# preprocess_data() 的内部主要会做：
# 1. clean_data：清理缺失严重的股票数据。
# 2. add_technical_indicator：添加 MACD、RSI 等技术指标。
# 3. add_vix：添加 VIX。
# 4. add_turbulence：添加 turbulence。
# 5. ffill/bfill：前向/后向填充缺失值。
processed = fe.preprocess_data(df_raw)

# 取出所有出现过的股票代码，准备后面构造完整的面板数据。
list_ticker = processed["tic"].unique().tolist()
# 生成从最早日期到最晚日期的完整日期序列。
# pd.date_range 会按天生成日期，再转成字符串，和 processed["date"] 的格式保持一致。
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
# 生成“日期 x 股票”的所有组合。
# 例如某一天有 30 只股票，那么这一天就应该出现 30 行记录。
combination = list(itertools.product(list_date, list_ticker))

# 把完整组合转成 DataFrame，再和已有处理结果做左连接。
# 这样可以把某些日期缺失的股票行也补出来，形成规则的二维面板结构。
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
# 只保留 processed 中实际存在过的日期。
# 这是因为 pd.date_range 会生成自然日，其中可能包含非交易日。
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
# 再按日期和股票代码排序，保证后续环境读取时顺序稳定。
processed_full = processed_full.sort_values(["date", "tic"])
# 对补齐后产生的缺失值填 0。
# 在 FinRL 的很多示例里，这是构造完整状态矩阵的常见处理方式。
processed_full = processed_full.fillna(0)

print("\n=== Processed data ===")
# 这里看到的是“可直接喂给训练/交易流程”的特征化数据。
print(processed_full.head())

# %% Part 4. Split and save data

# 第 4 部分：按日期切出训练集和交易集。
# data_split(df, start, end) 的逻辑是：保留 start <= date < end 的数据。
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
# 输出样本量，帮助你确认切分范围是否符合预期。
print(f"\nTrain data length: {len(train)}")
print(f"Trade data length: {len(trade)}")

# 保存为 CSV，供后面的训练脚本和回测脚本复用。
# 这里写的是相对路径，所以文件会输出到当前工作目录。
train.to_csv("train_data.csv")
trade.to_csv("trade_data.csv")
print("Data saved to train_data.csv and trade_data.csv")
