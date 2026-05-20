"""
红利低波ETF (512890) RSI策略回测
策略：RSI(15) 动态阈值（波动率调整），或固定阈值 RSI(15) < 32 买入，> 77 卖出

注意：512890是累积型ETF，分红已自动再投资体现在价格中，无需额外处理分红
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
import json
import os

# ============ 配置参数 ============
ETF_CODE = "512890"
ETF_NAME = "红利低波ETF"
RSI_PERIOD = 15  # 优化后：15日（原14日）
RSI_BUY_THRESHOLD = 32  # 优化后：32（原66）
RSI_SELL_THRESHOLD = 77  # 优化后：77（原81）
INITIAL_CAPITAL = 100000  # 初始资金10万

# 动态阈值模式
USE_DYNAMIC_THRESHOLD = True
RSI_BUY_BASE = 34
RSI_SELL_BASE = 71
VOL_WINDOW = 55
K_VOL = -0.423847
VOL_ANCHOR = 15.0
BUY_THRESHOLD_MIN = 20
BUY_THRESHOLD_MAX = 50
SELL_THRESHOLD_MIN = 60
SELL_THRESHOLD_MAX = 90

# 基准ETF配置
BENCHMARK_ETFS = {
    'hs300': {'code': '510300', 'name': '沪深300ETF'},
    'gold': {'code': '518880', 'name': '黄金ETF'},
    'nasdaq': {'code': '159941', 'name': '纳指ETF'},
    'sp500': {'code': '513500', 'name': '标普500ETF'},
}


def load_previous_result(path):
    """Load previous backtest JSON for benchmark fallback when API fetch fails."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

# ============ RSI计算 ============
def calculate_rsi(prices, period=15):
    """计算RSI指标（使用EMA平滑，更敏感）"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 使用EMA而非SMA（更敏感，与优化脚本一致）
    alpha = 1 / period  # EMA平滑因子
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility(prices, window):
    """年化波动率（百分比）"""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252) * 100
    return vol


def calculate_dynamic_thresholds(volatility, buy_base, sell_base, k_vol, vol_anchor,
                                 buy_min, buy_max, sell_min, sell_max):
    """计算动态RSI阈值"""
    vol_diff = volatility - vol_anchor
    buy_threshold = (buy_base - k_vol * vol_diff).clip(buy_min, buy_max)
    sell_threshold = (sell_base + k_vol * vol_diff).clip(sell_min, sell_max)
    return buy_threshold, sell_threshold


# ============ 获取数据 ============
def get_etf_data(code):
    """获取ETF日线数据"""
    print(f"正在获取 {code} 历史数据...")
    try:
        # 获取ETF日线数据
        df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        })
        df = df.sort_values('date').reset_index(drop=True)
        print(f"获取到 {len(df)} 条数据，从 {df['date'].min()} 到 {df['date'].max()}")
        return df
    except Exception as e:
        print(f"获取ETF数据失败: {e}")
        return None


def get_benchmark_data(code, name, index_type="index"):
    """获取基准指数数据"""
    print(f"正在获取基准 {name} 数据...")
    try:
        if index_type == "index":
            # 国内指数
            df = ak.index_zh_a_hist(symbol=code, period="daily", start_date="20131201")
        elif index_type == "us":
            # 美股指数 - 纳指100
            df = ak.index_us_stock_sina(symbol=code)
            
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={
            '日期': 'date',
            '收盘': 'close'
        })
        df = df.sort_values('date').reset_index(drop=True)
        print(f"获取到 {name} {len(df)} 条数据")
        return df[['date', 'close']]
    except Exception as e:
        print(f"获取 {name} 数据失败: {e}")
        return None


# ============ 回测引擎 ============
def run_backtest(df, initial_capital=INITIAL_CAPITAL):
    """
    执行RSI策略回测
    
    注意：512890是累积型ETF，分红已自动再投资体现在前复权价格中
    返回：交易记录、每日净值
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)

    if USE_DYNAMIC_THRESHOLD:
        df['volatility'] = calculate_volatility(df['close'], VOL_WINDOW)
        df['buy_threshold'], df['sell_threshold'] = calculate_dynamic_thresholds(
            df['volatility'], RSI_BUY_BASE, RSI_SELL_BASE, K_VOL, VOL_ANCHOR,
            BUY_THRESHOLD_MIN, BUY_THRESHOLD_MAX, SELL_THRESHOLD_MIN, SELL_THRESHOLD_MAX
        )
    
    # 初始化
    cash = initial_capital
    shares = 0
    position = 0  # 0: 空仓, 1: 持仓
    
    trades = []  # 交易记录
    daily_values = []  # 每日净值
    
    for i, row in df.iterrows():
        date = row['date']
        price = row['close']
        rsi = row['rsi']
        date_str = date.strftime('%Y-%m-%d')
        
        # 确定当前阈值
        if USE_DYNAMIC_THRESHOLD:
            buy_thresh = row['buy_threshold'] if pd.notna(row.get('buy_threshold')) else None
            sell_thresh = row['sell_threshold'] if pd.notna(row.get('sell_threshold')) else None
        else:
            buy_thresh = RSI_BUY_THRESHOLD
            sell_thresh = RSI_SELL_THRESHOLD

        # RSI信号判断
        if pd.notna(rsi) and buy_thresh is not None and sell_thresh is not None:
            if rsi < buy_thresh and position == 0:
                # 买入信号：满仓买入
                shares_to_buy = int(cash / price / 100) * 100  # 整百份
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date_str,
                        'action': '买入',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': cost,
                        'rsi': rsi,
                        'total_shares': shares,
                        'cash': cash
                    })
                    
            elif rsi > sell_thresh and position == 1:
                # 卖出信号：全部卖出
                if shares > 0:
                    sell_shares = int(shares / 100) * 100  # 整百份
                    if sell_shares > 0:
                        revenue = sell_shares * price
                        cash += revenue
                        shares -= sell_shares
                        if shares < 100:
                            # 剩余零头也卖掉
                            cash += shares * price
                            shares = 0
                        position = 0
                        trades.append({
                            'date': date_str,
                            'action': '卖出',
                            'price': price,
                            'shares': sell_shares,
                            'amount': revenue,
                            'rsi': rsi,
                            'total_shares': shares,
                            'cash': cash
                        })
        
        # 计算当日总资产
        total_value = cash + shares * price
        daily_values.append({
            'date': date_str,
            'close': price,
            'rsi': rsi if pd.notna(rsi) else None,
            'volatility': float(row['volatility']) if USE_DYNAMIC_THRESHOLD and pd.notna(row.get('volatility')) else None,
            'buy_threshold': float(buy_thresh) if buy_thresh is not None else None,
            'sell_threshold': float(sell_thresh) if sell_thresh is not None else None,
            'cash': cash,
            'shares': shares,
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return trades, daily_values


def calculate_buy_and_hold(df, initial_capital=INITIAL_CAPITAL):
    """计算买入持有策略
    
    注意：512890是累积型ETF，分红已体现在前复权价格中
    """
    start_price = df.iloc[0]['close']
    shares = int(initial_capital / start_price / 100) * 100
    remaining_cash = initial_capital - shares * start_price
    
    daily_values = []
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        price = row['close']
        
        total_value = remaining_cash + shares * price
        daily_values.append({
            'date': date_str,
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return daily_values


def calculate_benchmark_return(df, initial_capital=INITIAL_CAPITAL, reference_dates=None):
    """计算基准收益
    
    Args:
        df: 基准数据DataFrame
        initial_capital: 初始资金
        reference_dates: 参考日期列表，用于对齐数据。如果提供，只返回这些日期的数据
    """
    if df is None or len(df) == 0:
        return []
    
    # 创建日期到价格的映射
    date_price_map = {}
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        date_price_map[date_str] = row['close']
    
    # 如果提供了参考日期，按参考日期对齐
    if reference_dates:
        start_price = None
        daily_values = []
        
        for date_str in reference_dates:
            if date_str in date_price_map:
                price = date_price_map[date_str]
                if start_price is None:
                    start_price = price
                total_value = initial_capital * (price / start_price)
                daily_values.append({
                    'date': date_str,
                    'total_value': total_value,
                    'return': (total_value / initial_capital - 1) * 100
                })
            # 如果日期不存在，使用前一个值（向前填充）
            elif daily_values:
                daily_values.append({
                    'date': date_str,
                    'total_value': daily_values[-1]['total_value'],
                    'return': daily_values[-1]['return']
                })
        
        return daily_values
    
    # 原始逻辑
    start_price = df.iloc[0]['close']
    daily_values = []
    
    for _, row in df.iterrows():
        price = row['close']
        total_value = initial_capital * (price / start_price)
        daily_values.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return daily_values


def calculate_statistics(daily_values, trades):
    """计算策略统计指标"""
    if not daily_values:
        return {}
    
    returns = [d['return'] for d in daily_values]
    values = [d['total_value'] for d in daily_values]
    
    # 计算最大回撤
    peak = values[0]
    max_drawdown = 0
    for v in values:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 计算年化收益（使用自然日天数，而非交易日数）
    trading_days = len(daily_values)
    total_return = returns[-1]
    # 计算起止日期的自然天数
    from datetime import datetime
    start_date = datetime.strptime(daily_values[0]['date'], '%Y-%m-%d')
    end_date = datetime.strptime(daily_values[-1]['date'], '%Y-%m-%d')
    calendar_days = (end_date - start_date).days
    annual_return = ((1 + total_return / 100) ** (365 / calendar_days) - 1) * 100 if calendar_days > 0 else 0
    
    # 交易统计
    buy_trades = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']
    
    # 计算胜率
    wins = 0
    for i, sell in enumerate(sell_trades):
        if i < len(buy_trades):
            if sell['price'] > buy_trades[i]['price']:
                wins += 1
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
    
    return {
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'trade_count': len(buy_trades),
        'win_rate': round(win_rate, 2),
        'start_date': daily_values[0]['date'],
        'end_date': daily_values[-1]['date'],
        'days': trading_days,
        'calendar_days': calendar_days
    }


def calculate_annual_return(total_return_pct, calendar_days):
    """计算复利年化收益率
    
    Args:
        total_return_pct: 总收益率百分比
        calendar_days: 自然日天数（非交易日）
    
    公式: annual_return = (1 + total_return) ^ (365/days) - 1
    """
    if calendar_days <= 0 or total_return_pct is None:
        return None
    return round(((1 + total_return_pct / 100) ** (365 / calendar_days) - 1) * 100, 2)


# ============ 主程序 ============
def main():
    print("=" * 60)
    mode = "动态阈值" if USE_DYNAMIC_THRESHOLD else "固定阈值"
    print("红利低波ETF (512890) RSI策略回测 [%s]" % mode)
    print("=" * 60)
    
    # 1. 获取数据
    etf_df = get_etf_data(ETF_CODE)
    if etf_df is None:
        print("无法获取ETF数据，退出")
        return
    
    # 2. 统一时间范围
    start_date = etf_df['date'].min()
    end_date = etf_df['date'].max()
    print(f"\n回测区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 3. 获取基准ETF数据
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "backtest_result.json")
    previous_result = load_previous_result(output_file)
    previous_stats = previous_result.get('statistics', {}) if previous_result else {}
    previous_daily = previous_result.get('daily_values', {}) if previous_result else {}

    benchmark_data = {}
    for key, info in BENCHMARK_ETFS.items():
        print(f"正在获取 {info['name']} ({info['code']}) 数据...")
        try:
            df = ak.fund_etf_hist_em(symbol=info['code'], period="daily", adjust="qfq")
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.rename(columns={'日期': 'date', '收盘': 'close'})
            df = df.sort_values('date').reset_index(drop=True)
            # 筛选到相同时间范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            benchmark_data[key] = df[['date', 'close']]
            print(f"  获取到 {len(df)} 条数据")
        except Exception as e:
            print(f"  获取失败: {e}")
            benchmark_data[key] = None
    
    # 4. 执行回测（无需分红处理，累积型ETF分红已体现在价格中）
    print("\n正在执行RSI策略回测...")
    trades, strategy_values = run_backtest(etf_df)
    
    print("正在计算买入持有收益...")
    buyhold_values = calculate_buy_and_hold(etf_df)
    
    print("正在计算基准收益...")
    
    # 获取策略的日期列表，用于对齐所有基准数据
    strategy_dates = [d['date'] for d in strategy_values]
    
    # 计算各基准收益（使用策略日期对齐）
    benchmark_values = {}
    benchmark_returns = {}
    for key, df in benchmark_data.items():
        if df is not None and len(df) > 0:
            values = calculate_benchmark_return(df, reference_dates=strategy_dates)
            benchmark_values[key] = values
            benchmark_returns[key] = round(values[-1]['return'], 2) if values else None
        else:
            fallback_values = previous_daily.get(key, [])
            fallback_return = previous_stats.get(f'{key}_return')
            benchmark_values[key] = fallback_values if fallback_values else []
            benchmark_returns[key] = fallback_return
            if fallback_return is not None:
                print(f"  {BENCHMARK_ETFS[key]['name']} 使用上次结果兜底: {fallback_return:.2f}%")
            else:
                print(f"  {BENCHMARK_ETFS[key]['name']} 无可用兜底数据")
    
    # 5. 计算统计指标
    strategy_stats = calculate_statistics(strategy_values, trades)
    buyhold_stats = calculate_statistics(buyhold_values, [])
    
    # 使用自然日天数计算年化收益率
    calendar_days = strategy_stats.get('calendar_days', strategy_stats['days'])
    
    # 计算各基准的年化收益率
    benchmark_annuals = {}
    for key in benchmark_returns:
        annual = calculate_annual_return(benchmark_returns.get(key), calendar_days)
        if annual is None:
            annual = previous_stats.get(f'{key}_annual')
        benchmark_annuals[key] = annual

    # 记录回测天数供导出使用
    backtest_days = strategy_stats.get('days', len(strategy_values))
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"\n【RSI策略】")
    print(f"  总收益率: {strategy_stats['total_return']:.2f}%")
    print(f"  年化收益: {strategy_stats['annual_return']:.2f}%")
    print(f"  最大回撤: {strategy_stats['max_drawdown']:.2f}%")
    print(f"  交易次数: {strategy_stats['trade_count']} 次")
    print(f"  胜率: {strategy_stats['win_rate']:.2f}%")
    
    print(f"\n【买入持有】")
    print(f"  总收益率: {buyhold_stats['total_return']:.2f}%")
    print(f"  年化收益: {buyhold_stats['annual_return']:.2f}%")
    
    for key, info in BENCHMARK_ETFS.items():
        if benchmark_returns.get(key) is not None:
            print(f"\n【{info['name']}】")
            print(f"  总收益率: {benchmark_returns[key]:.2f}%")
    
    # 6. 导出数据为JSON
    # 准备导出数据
    export_data = {
        'meta': {
            'etf_code': ETF_CODE,
            'etf_name': ETF_NAME,
            'strategy': 'RSI(%d) 动态阈值 [buy_base=%d, sell_base=%d, k_vol=%.6f, vol_window=%d]' % (
                RSI_PERIOD, RSI_BUY_BASE, RSI_SELL_BASE, K_VOL, VOL_WINDOW
            ) if USE_DYNAMIC_THRESHOLD else 'RSI(%d) < %d 买入, > %d 卖出' % (RSI_PERIOD, RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD),
            'initial_capital': INITIAL_CAPITAL,
            'start_date': strategy_stats['start_date'],
            'end_date': strategy_stats['end_date'],
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'statistics': {
            'strategy': strategy_stats,
            'buyhold': buyhold_stats,
            'hs300_return': benchmark_returns.get('hs300'),
            'hs300_annual': benchmark_annuals.get('hs300'),
            'gold_return': benchmark_returns.get('gold'),
            'gold_annual': benchmark_annuals.get('gold'),
            'nasdaq_return': benchmark_returns.get('nasdaq'),
            'nasdaq_annual': benchmark_annuals.get('nasdaq'),
            'sp500_return': benchmark_returns.get('sp500'),
            'sp500_annual': benchmark_annuals.get('sp500'),
            'backtest_days': backtest_days,
        },
        'trades': trades,
        'daily_values': {
            'strategy': strategy_values,
            'buyhold': buyhold_values,
            'hs300': benchmark_values.get('hs300', []),
            'gold': benchmark_values.get('gold', []),
            'nasdaq': benchmark_values.get('nasdaq', []),
            'sp500': benchmark_values.get('sp500', []),
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测结果已保存至: {output_file}")
    
    # 同时复制到docs目录供网页使用
    docs_output = os.path.join(os.path.dirname(output_dir), "docs", "backtest_result.json")
    with open(docs_output, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False)
    print(f"网页数据已保存至: {docs_output}")
    
    return export_data


if __name__ == "__main__":
    main()
