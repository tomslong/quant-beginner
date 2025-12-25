# 导入函数库
import warnings

import numpy as np
import pandas as pd
import talib
from jqdata import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings("ignore")


## 初始化函数，设定基准等等
def initialize(context):
    # 避免获得未来函数
    set_option("avoid_future_data", True)
    # 设定沪深300作为基准
    set_benchmark("000300.XSHG")
    # 开启动态复权模式(真实价格)
    set_option("use_real_price", True)
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    # 输出内容到日志 log.info()
    log.info("初始函数开始运行且全局只运行一次")

    ### 期货相关设定 ###
    # 设定账户为金融账户
    set_subportfolios(
        [SubPortfolioConfig(cash=context.portfolio.starting_cash, type="index_futures")]
    )
    # 期货类每笔交易时的手续费是：买入时万分之0.23,卖出时万分之0.23,平今仓为万分之23
    set_order_cost(
        OrderCost(
            open_commission=0.000023,
            close_commission=0.000023,
            close_today_commission=0.00023,
        ),
        type="index_futures",
    )
    # 设定保证金比例
    set_option("futures_margin_rate", 0.08)

    # 设置期货交易的滑点
    set_slippage(StepRelatedSlippage(2))

    # 全局变量
    g.current_contract = None # 当日主力合约
    g.model = None  # 随机森林模型
    g.trade_allowed = False # 当日是否运行交易（模型判断）
    g.prediction_direction = 0  # 预测方向：1=看涨, -1=看跌, 0=无明确方向
    
    
    # 运行函数
    run_daily(before_market_open, time="09:30", reference_security="IF9999.CCFX")
    run_daily(market_open, time="every_bar", reference_security="IF9999.CCFX")
    run_daily(close_all_positions, time="14:55", reference_security="IF9999.CCFX")
    
    train_model(context)



## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info("函数运行时间(before_market_open)：" + str(context.current_dt.time()))

    if_contracts = get_future_contracts('IF', date=context.current_dt)
    g.current_contract = if_contracts[0]
    if g.model is None:
        train_model(context)

    ## 模型预测
    # 使用已有的模型预测今日走势
    end_date = context.current_dt - datetime.timedelta(days=1)

    # 获取日线进行特征提取
    daily_data = get_price('000300.XSHG', 
                           count=100, 
                           end_date=end_date,
                           frequency='daily',
                           fields=['open','close','high','low'])
    
    if len(daily_data) > 30:
        # 准备昨天的特征输入给模型
        features = get_features(daily_data)
        if (features is not None) and (len(features)>0):
            last_features =features.iloc[-1:].values
            # 预测今日
            prediction = g.model.predict(last_features)[0]
            if prediction ==1:
                g.trade_allowed = True
                g.prediction_direction = 1
                log.info("》》模型预测：今日看涨（Close > Open),允许开仓多头")
            elif prediction ==-1:
                g.trade_allowed = True
                g.prediction_direction = -1
                log.info("》》模型预测：今日看跌（Close < Open),允许开仓空头")
            else:
                g.trade_allowed = False
                g.prediction_direction = 0
                log.info("》》模型预测：今日震荡，暂停开仓")
        else:
            g.trade_allowed = False
            g.prediction_direction = 0
            log.warning("特征提取后有效样本为0，跳过今日")
    else:
        g.trade_allowed = False
        g.prediction_direction = 0
        log.warning("历史数据不足，今日暂停交易")


## 开盘时运行函数
def market_open(context):
    current_time = context.current_dt.time()
    
    # 1. 风控：14:55 后停止开新仓，准备尾盘平仓
    if current_time >= datetime.time(14, 55):
        return 

    # ================= 增加交割日移仓逻辑 =================
    # 获取当月和下月合约
    if_contracts = get_future_contracts('IF', date=context.current_dt)
    
    # 确保获取到了合约
    if len(if_contracts) >= 2:
        this_month_contract = if_contracts[0]
        next_month_contract = if_contracts[1]
        
        # 获取当月合约的详细信息，查看end_date
        info = get_security_info(this_month_contract)
        current_date = context.current_dt.date()
        
        # 判断是否是交割日
        is_delivery_day = (info.end_date == current_date)
        
        if is_delivery_day:
            # 如果是交割日，我们的目标交易合约直接切换为“下月合约”
            g.current_contract = next_month_contract
            
            # 检查手里有没有旧合约的持仓，如果有，必须强制平掉
            if this_month_contract in context.portfolio.long_positions:
                old_position = context.portfolio.long_positions[this_month_contract]
                if old_position.total_amount > 0:
                    log.info(f"》》交割日移仓：强制平掉旧合约 {this_month_contract}，目标切换为 {g.current_contract}")
                    order_target(this_month_contract, 0, side='long')
        else:
            # 如果不是交割日，确保目标还是当月合约
            g.current_contract = this_month_contract
    
    # 2. 交易准入：如果是模型强烈看空，跳过（可选）
    if not g.trade_allowed:
        return

    # 3. 获取 10分钟 K线数据 (降频，减少噪音)
    # Count=80 确保足够计算 MA60 和其他指标
    prices = get_price('000300.XSHG', 
                       end_date=context.current_dt, 
                       count=80, 
                       frequency='3m', 
                       fields=['close', 'open', 'high', 'low']) 
    
    if prices is None or len(prices) < 70:
        return 

    close_arr = prices['close'].values
    current_price = close_arr[-1]
    
   
    
    # 4. 计算指标
    # 计算MACD指标（根据10分钟周期调整参数，保持相同时间跨度）
    macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=18, slowperiod=39, signalperiod=14)
    
    # 计算均线用于趋势过滤
    ma20 = talib.SMA(close_arr, timeperiod=20)[-1]  # 中期均线
    ma60 = talib.SMA(close_arr, timeperiod=60)[-1]  # 长期均线
    
    # 计算 ATR 用于动态止损
    atr = talib.ATR(prices['high'].values, prices['low'].values, close_arr, timeperiod=14)[-1]

    # 获取持仓
    current_long = 0
    current_short = 0
    long_avg_cost = 0
    short_avg_cost = 0
    
    # 多头持仓
    if g.current_contract in context.portfolio.long_positions:
        pos = context.portfolio.long_positions[g.current_contract]
        current_long = pos.total_amount
        long_avg_cost = pos.avg_cost
    
    # 空头持仓
    if g.current_contract in context.portfolio.short_positions:
        pos = context.portfolio.short_positions[g.current_contract]
        current_short = pos.total_amount
        short_avg_cost = pos.avg_cost

    # ================= 核心策略改进 =================
    
    # 【开仓条件】
    # 确保有足够的历史数据计算MACD
    if len(macd) >= 2:
        # 多头开仓条件：MACD金叉且价格在MA20和MA60之上（注重长期趋势）
        macd_gold_cross = (macd[-1] > macd_signal[-1]) and (macd[-2] <= macd_signal[-2])
        price_above_ma20 = current_price > ma20
        price_above_ma60 = current_price > ma60  # 使用长期均线过滤
        
        # 空头开仓条件：MACD死叉且价格在MA20和MA60之下（注重长期趋势）
        macd_dead_cross = (macd[-1] < macd_signal[-1]) and (macd[-2] >= macd_signal[-2])
        price_below_ma20 = current_price < ma20
        price_below_ma60 = current_price < ma60  # 使用长期均线过滤
        
        # 计算额外的趋势过滤条件
        # MA20在MA60上方（中期上升趋势）
        ma20_above_ma60 = ma20 > ma60
        # 短期均线多头排列（MA5 > MA10 > MA20）
        ma5 = talib.SMA(close_arr, timeperiod=5)[-1]
        ma10 = talib.SMA(close_arr, timeperiod=10)[-1]
        short_ma_bullish = (ma5 > ma10) and (ma10 > ma20)
        # 短期均线空头排列（MA5 < MA10 < MA20）
        short_ma_bearish = (ma5 < ma10) and (ma10 < ma20)
        
        # 多头开仓：模型预测看涨 + MACD金叉 + 价格在MA20和MA60之上 + 中期上升趋势
        if g.prediction_direction == 1 and macd_gold_cross and price_above_ma20 and price_above_ma60 and ma20_above_ma60:
            if current_long == 0:
                # 如果有空头持仓，先平掉
                if current_short > 0:
                    order_target(g.current_contract, 0, side='short')
                log.info(f"》》模型预测看涨且10分钟MACD金叉。开多头！{g.current_contract}")
                order_target(g.current_contract, 2, side='long') # 开2手多头
        
        # 空头开仓：模型预测看跌 + MACD死叉 + 价格在MA20和MA60之下 + 短期空头排列
        elif g.prediction_direction == -1 and macd_dead_cross and price_below_ma20 and price_below_ma60 and short_ma_bearish:
            if current_short == 0:
                # 如果有多头持仓，先平掉
                if current_long > 0:
                    order_target(g.current_contract, 0, side='long')
                log.info(f"》》模型预测看跌且10分钟MACD死叉。开空头！{g.current_contract}")
                order_target(g.current_contract, 2, side='short') # 开2手空头

    # 【多头平仓条件】
    if current_long > 0:
        # 1. 止盈：MACD死叉 (MACD线从上下穿信号线)
        if len(macd) >= 2:
            macd_dead_cross = (macd[-1] < macd_signal[-1]) and (macd[-2] >= macd_signal[-2])
            if macd_dead_cross:
                log.info("》》MACD死叉，多头止盈离场。")
                order_target(g.current_contract, 0, side='long')
            
        # 2. 动态止损 (ATR吊灯止损)
        # 如果价格跌破 "进场价 - 2倍ATR"，说明波动异常
        stop_price = long_avg_cost - (2.0 * atr)
        
        if current_price < stop_price:
            log.warning(f"！！多头触及ATR动态止损线 {stop_price:.1f}，强制平仓")
            order_target(g.current_contract, 0, side='long')
            
        # 3. 硬止损 (保底)
        elif current_price < long_avg_cost * 0.99: # 1% 硬止损
            order_target(g.current_contract, 0, side='long')
            
    # 【空头平仓条件】
    if current_short > 0:
        # 1. 止盈：MACD金叉 (MACD线从下上穿信号线)
        if len(macd) >= 2:
            macd_gold_cross = (macd[-1] > macd_signal[-1]) and (macd[-2] <= macd_signal[-2])
            if macd_gold_cross:
                log.info("》》MACD金叉，空头止盈离场。")
                order_target(g.current_contract, 0, side='short')
            
        # 2. 动态止损 (ATR吊灯止损)
        # 如果价格突破 "进场价 + 2倍ATR"，说明波动异常
        stop_price = short_avg_cost + (2.0 * atr)
        
        if current_price > stop_price:
            log.warning(f"！！空头触及ATR动态止损线 {stop_price:.1f}，强制平仓")
            order_target(g.current_contract, 0, side='short')
            
        # 3. 硬止损 (保底)
        elif current_price > short_avg_cost * 1.01: # 1% 硬止损
            order_target(g.current_contract, 0, side='short')

# 辅助函数：为了日志显示价格
def get_close_price(security, count, freq):
    p = get_price(security, count=count, frequency=freq, fields=['close'])
    if p is not None and len(p) > 0:
        return p['close'].iloc[-1]
    return 0
    
def close_all_positions(context):
    """
    每日14：55平当天开的仓，包括多头和空头
    """
    log.info("》》时间到14：55，强制平掉所有持仓")
    for security in list(context.portfolio.positions.keys()):
        # 平掉多头持仓
        if security in context.portfolio.long_positions:
            long_pos = context.portfolio.long_positions[security]
            if long_pos.total_amount > 0:
                order_target(security, 0, side='long')
        # 平掉空头持仓
        if security in context.portfolio.short_positions:
            short_pos = context.portfolio.short_positions[security]
            if short_pos.total_amount > 0:
                order_target(security, 0, side='short')

def train_model(context):
    """
    训练日线模型：预测Close>Open
    """
    log.info("正在训练日线趋势模型...")
    
    # 获取历史日线数据
    start_date = context.current_dt - datetime.timedelta(days=1000)
    end_date = context.current_dt - datetime.timedelta(days=1)
    
    data = get_price('000300.XSHG', 
        start_date=start_date, 
        end_date=end_date,
        frequency='daily',
        fields=['open','close','high','low'])
    
    if len(data) < 100:
        log.warning("训练数据不足")
        return 
        
    df = get_features(data)
    
    # 构建涨跌标签 - 加入趋势延续性条件
    df['next_open'] = data['open'].shift(-1)
    df['next_close'] = data['close'].shift(-1)
    df['prev_close'] = data['close'].shift(0)  # 当前天的收盘价
    
    # 1=看涨, 0=震荡, -1=看跌
    df['target'] = 0
    # 只有当收盘价高于开盘价且收盘价高于前一天收盘价时，才标记为看涨（增强趋势延续性）
    df.loc[(df['next_close'] > df['next_open']) & (df['next_close'] > df['prev_close']), 'target'] = 1
    # 只有当收盘价低于开盘价且收盘价低于前一天收盘价时，才标记为看跌（增强趋势延续性）
    df.loc[(df['next_close'] < df['next_open']) & (df['next_close'] < df['prev_close']), 'target'] = -1
    
    # 去除包含NaN的行
    df.dropna(inplace=True)
    
    # 特征与标签分离 - 排除用于构建标签的辅助列prev_close
    feature_cols = [c for c in df.columns if c not in ['target','next_open','next_close','prev_close']]
    
    X = df[feature_cols]
    y = df['target']
    
    # 定义更广泛的参数分布以提高模型复杂度
    param_dist = {
        'n_estimators': np.arange(100, 300, 20),  # 增加树的数量范围
        'max_depth': np.arange(10, 30, 2),        # 增加树深度范围
        'min_samples_leaf': np.arange(2, 15, 2),   # 调整叶节点样本数范围
        'min_samples_split': np.arange(5, 25, 3),  # 增加分裂样本数参数
        'max_features': ['auto', 'sqrt', 'log2'],  # 增加特征选择方式
        'criterion': ['gini', 'entropy']           # 增加分裂标准参数
    }
    
    # 改进的随机搜索设置
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=40,  # 增加迭代次数以搜索更多参数组合
        cv=5,       # 增加交叉验证折数以提高模型稳定性
        scoring='accuracy',
        random_state=42,
        n_jobs=-1   # 使用所有可用CPU核心加速
    )

    # 执行随机搜索
    random_search.fit(X, y)
    
    # 获取最佳参数
    best_params = random_search.best_params_
    log.info(f"最佳参数组合: {best_params}")
    
    # 使用最佳参数创建模型
    rf = random_search.best_estimator_
    g.model = rf
    
    log.info("模型训练完成")

def get_features(data):
    df = pd.DataFrame(index=data.index)
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    
    # 基础均线
    df['ma5'] = talib.SMA(close, timeperiod=5)    # 短期均线
    df['ma10'] = talib.SMA(close, timeperiod=10)  # 中期均线
    df['ma60'] = talib.SMA(close, timeperiod=60)  # 长期均线 (添加60日均线)
    
    # 动量指标
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['mom'] = talib.MOM(close, timeperiod=5) # 价格动量
    
    # 波动率指标 (ATR) 衡量市场活跃度
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    
    # 价格相对位置 (归一化)
    df['pos_ma20'] = (close - talib.SMA(close, timeperiod=20)) / talib.SMA(close, timeperiod=20)
    df['pos_ma60'] = (close - talib.SMA(close, timeperiod=60)) / talib.SMA(close, timeperiod=60)  # 相对于长期均线的位置
    
    # 均线趋势特征：短期均线在长期均线上方为1，否则为0
    df['short_above_long'] = (df['ma10'] > df['ma60']).astype(int)
    
    # 增强趋势特征
    # 中期均线趋势：MA20在MA60上方为1，否则为0
    df['ma20_above_ma60'] = (talib.SMA(close, timeperiod=20) > talib.SMA(close, timeperiod=60)).astype(int)
    # 价格在长期均线上方为1，否则为0
    df['price_above_ma60'] = (close > talib.SMA(close, timeperiod=60)).astype(int)
    # 价格在长期均线上方且MA5在MA10上方（短期上升趋势）
    df['short_term_up_trend'] = ((df['ma5'] > df['ma10']) & df['price_above_ma60']).astype(int)
    # 价格在长期均线下方且MA5在MA10下方（短期下降趋势）
    df['short_term_down_trend'] = ((df['ma5'] < df['ma10']) & (~df['price_above_ma60'])).astype(int)
    
    # 加上量价特征
    df['roc'] = talib.ROC(close, timeperiod=10) # 变动率
    
    df.dropna(inplace=True)
    return df

    