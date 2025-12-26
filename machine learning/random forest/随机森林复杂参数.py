# 导入函数库
import warnings
import numpy as np
import pandas as pd
import talib
from jqdata import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")

## 初始化函数
def initialize(context):
    set_option("avoid_future_data", True)
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    log.info("趋势波段增强版策略启动")

    ### 期货相关设定 ###
    set_subportfolios([SubPortfolioConfig(cash=context.portfolio.starting_cash, type="index_futures")])
    set_order_cost(OrderCost(open_commission=0.000023, close_commission=0.000023, close_today_commission=0.00023), type="index_futures")
    set_option("futures_margin_rate", 0.08)
    set_slippage(StepRelatedSlippage(2))

    # 全局变量
    g.current_contract = None 
    g.model = None  
    g.prediction_direction = 0  # 1=看涨, -1=看跌
    
    # 止损参数
    g.trailing_stop_long = 0
    g.trailing_stop_short = 999999
    
    # 运行函数
    # 每天开盘前训练或更新预测
    run_daily(before_market_open, time="09:10", reference_security="IF9999.CCFX")
    # 交易逻辑：开盘后尽早运行，捕捉趋势
    run_daily(trade_logic, time="09:35", reference_security="IF9999.CCFX")
    # 尾盘检查（可选，用于防止极端风险）
    run_daily(check_risk, time="14:50", reference_security="IF9999.CCFX")

## 开盘前运行：训练与预测
def before_market_open(context):
    if_contracts = get_future_contracts('IF', date=context.current_dt)
    if len(if_contracts) > 0:
        g.current_contract = if_contracts[0]
    else:
        g.current_contract = 'IF9999.CCFX'

    # 1. 训练模型 (为了效率，建议每月或每周重新训练一次，这里保持每日训练以适应市场)
    # 如果觉得回测太慢，可以加逻辑判断：如果 g.model 不为空且日期不是周一，则跳过
    train_model(context)

    # 2. 模型预测
    predict_today(context)

def predict_today(context):
    if g.model is None:
        return

    end_date = context.current_dt - datetime.timedelta(days=1)
    # 获取数据用于提取昨天的特征
    daily_data = get_price('000300.XSHG', count=80, end_date=end_date, frequency='daily', fields=['open','close','high','low','volume'])
    
    if len(daily_data) > 60:
        features = get_features(daily_data)
        if (features is not None) and (len(features) > 0):
            last_x = features[g.feature_columns].iloc[-1:].values
            # 预测
            prediction = g.model.predict(last_x)[0]
            g.prediction_direction = prediction
            log.info(f"今日模型预测方向: {prediction} (1=多, -1=空, 0=震荡)")
        else:
            g.prediction_direction = 0
    else:
        g.prediction_direction = 0

## 核心交易逻辑
def trade_logic(context):
    if g.current_contract is None:
        return

    # 移仓换月处理
    check_and_switch_contract(context)
    
    current_contract = g.current_contract
    
    # 获取当前价格数据
    prices = get_price('000300.XSHG', end_date=context.current_dt, count=100, frequency='30m', fields=['close', 'high', 'low'])
    if len(prices) < 60:
        return
        
    close_arr = prices['close'].values
    high_arr = prices['high'].values
    low_arr = prices['low'].values
    current_price = close_arr[-1]
    
    # 计算 ATR 用于移动止损
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=20)[-1]
    
    # 计算趋势均线
    ma20 = talib.SMA(close_arr, timeperiod=20)[-1]
    ma60 = talib.SMA(close_arr, timeperiod=60)[-1] # 长期趋势线

    # 获取当前持仓
    long_pos = context.portfolio.long_positions[current_contract]
    short_pos = context.portfolio.short_positions[current_contract]
    
    has_long = long_pos.total_amount > 0
    has_short = short_pos.total_amount > 0

    # ================= 1. 止损/止盈逻辑 (优先执行) =================
    
    if has_long:
        # 更新最高价用于移动止损 (吊灯止损)
        # 如果当前价格创新高，上调止损线
        estimated_entry = long_pos.avg_cost
        # 初始止损：成本价 - 2ATR
        # 移动止损：最高价回撤 2.5ATR
        
        # 简单的移动止损逻辑：
        # 如果是新开仓，初始化止损
        if g.trailing_stop_long == 0:
            g.trailing_stop_long = estimated_entry - 2.0 * atr
        
        # 随着价格上涨，提升止损线，只升不降
        new_stop = current_price - 2.5 * atr
        if new_stop > g.trailing_stop_long:
            g.trailing_stop_long = new_stop
            
        # 触发止损
        if current_price < g.trailing_stop_long:
            log.info(f"触发移动止损/止盈，平多头。当前价: {current_price}, 止损线: {g.trailing_stop_long}")
            order_target(current_contract, 0, side='long')
            g.trailing_stop_long = 0
            has_long = False
            
        # 趋势反转平仓：跌破MA60 且 模型不再看涨
        if current_price < ma60 and g.prediction_direction != 1:
            log.info("趋势反转(跌破MA60)，平多头")
            order_target(current_contract, 0, side='long')
            has_long = False

    if has_short:
        if g.trailing_stop_short == 999999:
            g.trailing_stop_short = short_pos.avg_cost + 2.0 * atr
            
        new_stop = current_price + 2.5 * atr
        if new_stop < g.trailing_stop_short:
            g.trailing_stop_short = new_stop
            
        if current_price > g.trailing_stop_short:
            log.info(f"触发移动止损/止盈，平空头。当前价: {current_price}, 止损线: {g.trailing_stop_short}")
            order_target(current_contract, 0, side='short')
            g.trailing_stop_short = 999999
            has_short = False
            
        if current_price > ma60 and g.prediction_direction != -1:
            log.info("趋势反转(升破MA60)，平空头")
            order_target(current_contract, 0, side='short')
            has_short = False

    # ================= 2. 开仓逻辑 (趋势跟随) =================
    # 不再等待MACD金叉，而是看重“位置”和“模型方向”
    
    # 开多条件：模型看涨 AND 价格在MA60之上 (多头区域) AND 没有持仓
    if (not has_long) and (not has_short):
        if g.prediction_direction == 1 and current_price > ma60:
            log.info("模型看涨且价格位于MA60之上，开仓做多，持有过夜")
            order_target(current_contract, 2, side='long')
            # 重置止损
            g.trailing_stop_long = current_price - 2.0 * atr
            
        # 开空条件：模型看跌 AND 价格在MA60之下 (空头区域)
        elif g.prediction_direction == -1 and current_price < ma60:
            log.info("模型看跌且价格位于MA60之下，开仓做空，持有过夜")
            order_target(current_contract, 2, side='short')
            # 重置止损
            g.trailing_stop_short = current_price + 2.0 * atr

def check_and_switch_contract(context):
    # 处理交割日移仓
    if_contracts = get_future_contracts('IF', date=context.current_dt)
    if len(if_contracts) < 2: return
    
    this_month = if_contracts[0]
    next_month = if_contracts[1]
    
    info = get_security_info(this_month)
    if info.end_date == context.current_dt.date():
        log.info("交割日，强制移仓到下月合约")
        g.current_contract = next_month
        
        # 平掉旧合约
        if context.portfolio.long_positions[this_month].total_amount > 0:
            order_target(this_month, 0, side='long')
            # 在新合约开同样的仓位(简单处理，立即开)
            order_target(next_month, 2, side='long')
            
        if context.portfolio.short_positions[this_month].total_amount > 0:
            order_target(this_month, 0, side='short')
            order_target(next_month, 2, side='short')

def check_risk(context):
    # 简单的尾盘风控，如果亏损幅度过大可以强平，否则保持持仓
    pass

def train_model(context):
    # 增加训练数据量
    start_date = context.current_dt - datetime.timedelta(days=1500)
    end_date = context.current_dt - datetime.timedelta(days=1)
    
    data = get_price('000300.XSHG', start_date=start_date, end_date=end_date, frequency='daily', fields=['open','close','high','low','volume'])
    if len(data) < 200: return
        
    df = get_features(data)
    
    # === 优化目标函数 ===
    # 原策略问题：要求Next Close > Next Open，这忽略了跳空高开的影响
    # 新策略目标：预测未来1天的回报率是否大于0（捕捉总体涨跌，不管是不是假阴线）
    df['ret_1d'] = df['close'].shift(-1) / df['close'] - 1.0
    
    # 标记：涨幅>0.1%为1，跌幅>0.1%为-1，其余为0
    # 降低噪音，只抓明显波动
    df['target'] = 0
    df.loc[df['ret_1d'] > 0.001, 'target'] = 1
    df.loc[df['ret_1d'] < -0.001, 'target'] = -1
    
    df.dropna(inplace=True)
    
    exclude_cols = ['target', 'ret_1d', 'close']
    g.feature_columns = [c for c in df.columns if c not in exclude_cols]
    X = df[g.feature_columns]
    y = df['target']
    
    # 简化随机搜索，提高运行速度
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced']
    }
    
    # 为了回测速度，这里减少n_iter
    rf = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10, 
        cv=3,
        random_state=42,
        n_jobs=1 
    )
    
    rf.fit(X, y)
    g.model = rf.best_estimator_

def get_features(data):
    df = pd.DataFrame(index=data.index)
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    volume = data['volume'].values if 'volume' in data.columns else np.zeros(len(close)) # 兼容无Volume情况
    
    df['close'] = close
    # 基础均线
    df['ma5'] = talib.SMA(close, timeperiod=5)
    df['ma20'] = talib.SMA(close, timeperiod=20)
    df['ma60'] = talib.SMA(close, timeperiod=60)
    
    # 动量与震荡
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    
    # 波动率 (重要特征：低波动通常预示变盘)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['atr_ratio'] = df['atr'] / close
    
    # 均线乖离率 (Bias)
    df['bias20'] = (close - df['ma20']) / df['ma20']
    
    # 相对位置
    df['price_pos'] = (close - talib.MIN(low, timeperiod=20)) / (talib.MAX(high, timeperiod=20) - talib.MIN(low, timeperiod=20) + 1e-9)
    
    # 新增特征：均线多头排列得分
    # 简单量化：短>中>长 得分最高
    condition1 = (df['ma5'] > df['ma20']).astype(int)
    condition2 = (df['ma20'] > df['ma60']).astype(int)
    df['trend_score'] = condition1 + condition2 
    
    # 量价关系 (如果有Volume)
    if 'volume' in data.columns:
        df['vol_ma5'] = talib.SMA(volume, timeperiod=5)
        df['vol_ratio'] = volume / (df['vol_ma5'] + 1e-9)

    df.dropna(inplace=True)
    return df