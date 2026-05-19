import os
import json
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

from data_fetcher import update_etf_data

# ==========================================
# 配置
# ==========================================
ETF_CODE = "512890"
ETF_NAME = "红利低波ETF"

BEST_PARAMS_PATH = os.path.join("backtest", "best_combined_params.json")
BACKTEST_RESULT_PATH = os.path.join("backtest", "backtest_result.json")

FEISHU_WEBHOOK = os.environ.get(
    "FEISHU_WEBHOOK",
    "https://open.feishu.cn/open-apis/bot/v2/hook/3efc8c66-ae0e-4f08-881a-670cc3d16681",
)


# ==========================================
# 工具函数
# ==========================================

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("read %s failed: %s" % (file_path, e))
        return None


def load_backtest_summary():
    data = load_json_file(BACKTEST_RESULT_PATH)
    stats = (data or {}).get("statistics", {})
    strategy_ideal = stats.get("strategy_ideal") or stats.get("strategy") or {}
    strategy_dynamic = stats.get("strategy_dynamic") or {}
    return {
        "classic_total": strategy_ideal.get("total_return"),
        "classic_annual": strategy_ideal.get("annual_return"),
        "dynamic_total": strategy_dynamic.get("total_return"),
        "dynamic_annual": strategy_dynamic.get("annual_return"),
    }


def load_dynamic_params():
    params = load_json_file(BEST_PARAMS_PATH) or {}
    return {
        "rsi_period": int(params.get("rsi_period", 15)),
        "rsi_buy_base": float(params.get("rsi_buy_base", 34)),
        "rsi_sell_base": float(params.get("rsi_sell_base", 72)),
        "vol_window": int(params.get("vol_window", 20)),
        "k_vol": float(params.get("k_vol", 0.0)),
        "vol_anchor": float(params.get("vol_anchor", 15.0)),
    }


# ==========================================
# RSI + 波动率 计算
# ==========================================

def calculate_rsi_ema(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_volatility_annualized(close_series, window):
    if close_series is None or len(close_series) < window + 1:
        return None
    log_ret = np.log(close_series / close_series.shift(1))
    vol = log_ret.rolling(window=window).std() * np.sqrt(252) * 100
    latest = vol.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)


def compute_dynamic_signal(rsi_value, close_series, params):
    vol = calculate_volatility_annualized(close_series, params["vol_window"])
    if vol is None or rsi_value is None:
        return None

    adjustment = params["k_vol"] * (vol - params["vol_anchor"])
    buy_threshold = min(50.0, max(20.0, params["rsi_buy_base"] - adjustment))
    sell_threshold = min(90.0, max(60.0, params["rsi_sell_base"] + adjustment))

    if rsi_value < buy_threshold:
        signal = "买入"
        signal_color = "#22c55e"
    elif rsi_value > sell_threshold:
        signal = "卖出"
        signal_color = "#ef4444"
    else:
        signal = "仓位不动"
        signal_color = "#3b82f6"

    return {
        "volatility": round(vol, 2),
        "buy_threshold": round(buy_threshold, 2),
        "sell_threshold": round(sell_threshold, 2),
        "signal": signal,
        "signal_color": signal_color,
    }


# ==========================================
# 飞书通知
# ==========================================

def send_feishu(title, content):
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": [
                {"tag": "markdown", "content": content},
            ],
        },
    }
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        print("feishu response: %s" % resp.text)
    except Exception as e:
        print("feishu send failed: %s" % e)


# ==========================================
# 数据获取（增量 + 多源）
# ==========================================

def fetch_rsi_and_price(params):
    print("[%s] fetching data..." % datetime.now().strftime("%H:%M:%S"))

    df = update_etf_data(ETF_CODE, max_rows=300)

    rsi_period = params["rsi_period"]
    if df is None or len(df) < rsi_period + 5:
        print("insufficient data")
        return None, None, None, None

    # 只用最近60行计算RSI，与原策略保持一致
    df_rsi = df.tail(60).reset_index(drop=True)

    df_rsi["rsi"] = calculate_rsi_ema(df_rsi["close"], rsi_period)

    latest = df_rsi.iloc[-1]
    rsi_value = latest["rsi"]
    latest_price = latest["close"]
    latest_date = latest["date"].strftime("%Y-%m-%d")

    if pd.isna(rsi_value):
        print("RSI calc failed")
        return None, None, None, None

    print("RSI(%d) = %.2f | price = %.4f | date = %s" % (
        rsi_period, rsi_value, latest_price, latest_date))
    return rsi_value, latest_price, latest_date, df_rsi


# ==========================================
# 主逻辑
# ==========================================

def is_wednesday_beijing():
    """判断当前北京时间是否为周三"""
    beijing_now = datetime.utcnow() + timedelta(hours=8)
    return beijing_now.weekday() == 2


def main():
    dynamic_params = load_dynamic_params()
    rsi, price, latest_date, market_df = fetch_rsi_and_price(dynamic_params)

    if rsi is None:
        print("no valid RSI data, exit.")
        return

    dynamic_signal = compute_dynamic_signal(
        rsi, market_df["close"] if market_df is not None else None, dynamic_params
    )

    if dynamic_signal is None:
        print("dynamic signal computation failed, exit.")
        return

    signal = dynamic_signal["signal"]
    vol = dynamic_signal["volatility"]
    buy_th = dynamic_signal["buy_threshold"]
    sell_th = dynamic_signal["sell_threshold"]

    print("signal=%s | vol=%.2f | buy_th=%.2f | sell_th=%.2f" % (
        signal, vol, buy_th, sell_th))

    backtest_summary = load_backtest_summary()
    bt_return = "--"
    bt_annual = "--"
    if backtest_summary["dynamic_total"] is not None:
        bt_return = "%.2f%%" % backtest_summary["dynamic_total"]
    if backtest_summary["dynamic_annual"] is not None:
        bt_annual = "%.2f%%" % backtest_summary["dynamic_annual"]

    # --- 通知逻辑 ---
    if signal == "买入":
        title = "【买入信号】%s RSI=%.2f < %.2f" % (ETF_NAME, rsi, buy_th)
        content = (
            "**%s (%s)**\n\n"
            "RSI(%d) = **%.2f** < 动态买入阈值 **%.2f**\n\n"
            "波动率: %.2f%% | 卖出阈值: %.2f\n\n"
            "当前价格: **%.4f** | 数据日期: %s\n\n"
            "回测总收益: %s | 年化: %s"
        ) % (ETF_NAME, ETF_CODE, dynamic_params["rsi_period"], rsi, buy_th,
             vol, sell_th, price, latest_date, bt_return, bt_annual)
        send_feishu(title, content)

    elif signal == "卖出":
        title = "【卖出信号】%s RSI=%.2f > %.2f" % (ETF_NAME, rsi, sell_th)
        content = (
            "**%s (%s)**\n\n"
            "RSI(%d) = **%.2f** > 动态卖出阈值 **%.2f**\n\n"
            "波动率: %.2f%% | 买入阈值: %.2f\n\n"
            "当前价格: **%.4f** | 数据日期: %s\n\n"
            "回测总收益: %s | 年化: %s"
        ) % (ETF_NAME, ETF_CODE, dynamic_params["rsi_period"], rsi, sell_th,
             vol, buy_th, price, latest_date, bt_return, bt_annual)
        send_feishu(title, content)

    elif is_wednesday_beijing():
        title = "【心跳】%s 策略正常运行" % ETF_NAME
        content = (
            "RSI(%d) = %.2f | 动态买入阈值 %.2f | 动态卖出阈值 %.2f | 波动率 %.2f%%\n\n"
            "当前信号: **仓位不动** | 价格: %.4f | 数据日期: %s"
        ) % (dynamic_params["rsi_period"], rsi, buy_th, sell_th, vol,
             price, latest_date)
        send_feishu(title, content)

    else:
        print("signal=仓位不动, not Wednesday, skip notification.")

    # --- 生成 docs JSON (保留给 GitHub Pages) ---
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    beijing_time = datetime.utcnow() + timedelta(hours=8)
    timestamp = beijing_time.strftime("%Y-%m-%d %H:%M:%S") + " (北京时间)"

    # === 生成 monitor.json（运行监控数据）===
    # 通知决策逻辑与上面保持一致
    notification_decision = "skip"
    notification_reason = ""

    if signal == "买入":
        notification_decision = "send"
        notification_reason = "RSI低于动态买入阈值，触发买入信号"
    elif signal == "卖出":
        notification_decision = "send"
        notification_reason = "RSI高于动态卖出阈值，触发卖出信号"
    elif is_wednesday_beijing():
        notification_decision = "send"
        notification_reason = "周三心跳包，策略正常运行"
    else:
        notification_decision = "skip"
        notification_reason = "信号为仓位不动，且非周三，不发送通知"

    # 读取历史记录
    monitor_json_path = os.path.join(docs_dir, "monitor.json")
    try:
        with open(monitor_json_path, "r", encoding="utf-8") as f:
            monitor_history = json.load(f)
    except:
        monitor_history = {"runs": []}

    # 添加本次运行记录
    run_record = {
        "timestamp": timestamp,
        "rsi": round(rsi, 2),
        "price": round(price, 4) if price else None,
        "market_date": latest_date,
        "signal": signal,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "volatility": vol,
        "notification_decision": notification_decision,
        "notification_reason": notification_reason,
    }
    monitor_history["runs"].insert(0, run_record)

    # 保留最近 30 条记录
    monitor_history["runs"] = monitor_history["runs"][:30]

    with open(monitor_json_path, "w", encoding="utf-8") as f:
        json.dump(monitor_history, f, ensure_ascii=False, indent=2)

    # === 生成 data.json（前端展示用）===
    data_json = {
        "etf_code": ETF_CODE,
        "etf_name": ETF_NAME,
        "rsi": round(rsi, 2),
        "rsi_period": dynamic_params["rsi_period"],
        "market_date": latest_date,
        "price": round(price, 4) if price else None,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "signal": signal,
        "signal_color": dynamic_signal["signal_color"],
        "strategy": "RSI(%d) + Vol dynamic" % dynamic_params["rsi_period"],
        "backtest_return": bt_return,
        "backtest_annual": bt_annual,
        "timestamp": timestamp,
    }

    with open(os.path.join(docs_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    dynamic_data_json = {
        "etf_code": ETF_CODE,
        "etf_name": ETF_NAME,
        "market_date": latest_date,
        "price": round(price, 4) if price else None,
        "rsi": round(rsi, 2),
        "rsi_period": dynamic_params["rsi_period"],
        "rsi_buy_base": dynamic_params["rsi_buy_base"],
        "rsi_sell_base": dynamic_params["rsi_sell_base"],
        "vol_window": dynamic_params["vol_window"],
        "k_vol": dynamic_params["k_vol"],
        "vol_anchor": dynamic_params["vol_anchor"],
        "volatility": vol,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "signal": signal,
        "signal_color": dynamic_signal["signal_color"],
        "backtest_return": bt_return,
        "backtest_annual": bt_annual,
        "timestamp": timestamp,
    }

    with open(os.path.join(docs_dir, "dynamic_data.json"), "w", encoding="utf-8") as f:
        json.dump(dynamic_data_json, f, ensure_ascii=False, indent=2)

    print("docs JSON updated.")


if __name__ == "__main__":
    main()
