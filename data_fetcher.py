"""
多数据源 ETF 日线数据获取模块

优先级: 腾讯财经 > 新浪财经 > akshare(东方财富)
返回统一格式 DataFrame: [date, open, high, low, close, volume]
"""

import os
import json
from datetime import datetime, timedelta

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# 代码格式转换
# ---------------------------------------------------------------------------

def _code_to_tencent(code):
    """512890 -> sh512890"""
    if code.startswith(("5", "6", "11")):
        return "sh" + code
    return "sz" + code


def _code_to_sina(code):
    """同腾讯格式"""
    return _code_to_tencent(code)


# ---------------------------------------------------------------------------
# 腾讯财经 (主)
# ---------------------------------------------------------------------------

def fetch_tencent(code, start_date=None, count=300):
    """
    腾讯财经日线前复权数据
    http://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    """
    tc_code = _code_to_tencent(code)
    start_str = start_date if start_date else "2020-01-01"
    end_str = datetime.now().strftime("%Y-%m-%d")

    url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": "{code},day,{start},{end},{count},qfq".format(
        code=tc_code, start=start_str, end=end_str, count=count)}
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    inner = data.get("data", {}).get(tc_code, {})
    klines = inner.get("qfqday") or inner.get("day")
    if not klines:
        return None

    rows = []
    for k in klines:
        rows.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]) if len(k) > 5 else 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print("[Tencent] fetched %d rows for %s" % (len(df), code))
    return df


# ---------------------------------------------------------------------------
# 新浪财经 (备)
# ---------------------------------------------------------------------------

def fetch_sina(code, start_date=None, count=300):
    """
    新浪财经日线数据（不支持日期范围，只返回最近 N 条）
    """
    sina_code = _code_to_sina(code)
    url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {
        "symbol": sina_code,
        "scale": "240",
        "ma": "no",
        "datalen": str(count),
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None

    rows = []
    for item in data:
        rows.append({
            "date": item["day"],
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
            "volume": float(item["volume"]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

    print("[Sina] fetched %d rows for %s" % (len(df), code))
    return df


# ---------------------------------------------------------------------------
# akshare / 东方财富 (末)
# ---------------------------------------------------------------------------

def fetch_akshare(code, start_date=None, count=300):
    """
    akshare 底层调用东方财富 API，间歇性 502。
    """
    import akshare as ak

    kwargs = {"symbol": code, "period": "daily", "adjust": "qfq"}
    if start_date:
        kwargs["start_date"] = start_date.replace("-", "")
    kwargs["end_date"] = datetime.now().strftime("%Y%m%d")

    df = ak.fund_etf_hist_em(**kwargs)
    if df is None or len(df) == 0:
        return None

    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)

    if count and len(df) > count:
        df = df.tail(count).reset_index(drop=True)

    print("[akshare] fetched %d rows for %s" % (len(df), code))
    return df


# ---------------------------------------------------------------------------
# 统一接口：多源回退
# ---------------------------------------------------------------------------

_SOURCES = [
    ("Tencent", fetch_tencent),
    ("Sina", fetch_sina),
    ("akshare", fetch_akshare),
]


def fetch_etf_daily(code, start_date=None, count=300):
    """
    多源回退获取 ETF 日线数据。
    返回 DataFrame[date, open, high, low, close, volume]，按 date 升序。
    """
    errors = []
    for name, func in _SOURCES:
        try:
            df = func(code, start_date=start_date, count=count)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print("[%s] failed: %s" % (name, e))
            errors.append((name, e))

    raise RuntimeError("all data sources failed: %s" % errors)


# ---------------------------------------------------------------------------
# 增量更新 + CSV 持久化
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _csv_path(code):
    return os.path.join(DATA_DIR, "%s_daily.csv" % code)


def load_local_data(code):
    """读取本地 CSV，返回 DataFrame 或 None"""
    path = _csv_path(code)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_local_data(code, df, max_rows=300):
    """保存到本地 CSV，只保留最近 max_rows 行"""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    df.to_csv(_csv_path(code), index=False)
    print("[Save] %s -> %d rows" % (_csv_path(code), len(df)))


def update_etf_data(code, max_rows=300):
    """
    增量更新：读取本地 CSV -> 只拉增量 -> 合并去重 -> 保存。
    首次运行拉取全量。返回合并后的 DataFrame。
    """
    local_df = load_local_data(code)

    if local_df is not None and len(local_df) > 0:
        last_date = local_df["date"].max()
        start = (last_date - timedelta(days=3)).strftime("%Y-%m-%d")
        print("[Update] incremental from %s" % start)
        new_df = fetch_etf_daily(code, start_date=start, count=max_rows)
        merged = pd.concat([local_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        print("[Update] full fetch for %s" % code)
        merged = fetch_etf_daily(code, count=max_rows)

    save_local_data(code, merged, max_rows=max_rows)
    return merged


# ---------------------------------------------------------------------------
# 独立运行测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    code = "512890"
    print("=== Testing multi-source fetch for %s ===" % code)
    df = update_etf_data(code)
    print("\nResult: %d rows, from %s to %s" % (
        len(df), df["date"].min().date(), df["date"].max().date()))
    print(df.tail(5).to_string(index=False))
