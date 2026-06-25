#!/usr/bin/env python3
"""
tech_indicators.py — 技术面确定性计算引擎（供 /trade-signal skill 调用）

设计原则（与 financial_rigor.py 一致）：
  指标计算由 Python 完成，禁止 LLM 心算；LLM 只负责解读结果、下判断。

数据源：Yahoo Finance Chart API（通过 curl，绕过 Python SSL 问题，与 stock_screener.py 一致）

用法：
  python3 tech_indicators.py analyze NVDA              # 单标的技术面快照 + 信号旗标 + 风控建议
  python3 tech_indicators.py analyze 0700.HK           # 港股加 .HK；A股加 .SS/.SZ
  python3 tech_indicators.py analyze NVDA --json       # 输出原始 JSON
  python3 tech_indicators.py backtest NVDA             # 在该股历史上回测经典信号，给胜率/盈亏比

指标：MA(5/10/20/50/200) / RSI(14) / MACD(12,26,9) / KDJ(9,3,3) / 布林(20,2) / ATR(14) / OBV / 量比
所有指标为纯标准库实现，无 pandas/numpy 依赖。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime


# ============================================================
# 数据获取
# ============================================================

def fetch_ohlcv(ticker, rng="2y", interval="1d"):
    """用 curl 拉 Yahoo Finance OHLCV 日线。返回按时间升序的 list[dict]。"""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={rng}&interval={interval}"
    )
    try:
        result = subprocess.run(
            ["curl", "-s", "-H", "User-Agent: Mozilla/5.0", url],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return None, "curl 失败"
        data = json.loads(result.stdout)
        chart = data.get("chart", {})
        if chart.get("error"):
            return None, str(chart["error"])
        res = chart.get("result", [{}])[0]
        meta = res.get("meta", {})
        ts = res.get("timestamp", [])
        q = res.get("indicators", {}).get("quote", [{}])[0]
        rows = []
        for i, t in enumerate(ts):
            o, h, l, c, v = (q.get(k, [None] * len(ts))[i]
                             for k in ("open", "high", "low", "close", "volume"))
            if None in (o, h, l, c) or v is None:
                continue
            rows.append({
                "date": datetime.fromtimestamp(t).strftime("%Y-%m-%d"),
                "open": o, "high": h, "low": l, "close": c, "volume": v,
            })
        if len(rows) < 30:
            return None, f"数据不足（仅 {len(rows)} 根 K 线）"
        return {"rows": rows, "currency": meta.get("currency", "?"),
                "exchange": meta.get("fullExchangeName", "?")}, None
    except Exception as e:
        return None, f"异常: {e}"


# ============================================================
# 基础序列函数（纯标准库）
# ============================================================

def sma(vals, n):
    out = [None] * len(vals)
    for i in range(n - 1, len(vals)):
        out[i] = sum(vals[i - n + 1:i + 1]) / n
    return out


def ema(vals, n):
    out = [None] * len(vals)
    if len(vals) < n:
        return out
    k = 2 / (n + 1)
    seed = sum(vals[:n]) / n  # 用前 n 个的 SMA 作种子
    out[n - 1] = seed
    for i in range(n, len(vals)):
        out[i] = vals[i] * k + out[i - 1] * (1 - k)
    return out


def stddev_pop(vals):
    m = sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def rsi(closes, n=14):
    """Wilder 平滑 RSI。"""
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    def calc(g, l):
        if l == 0:
            return 100.0
        rs = g / l
        return 100 - 100 / (1 + rs)
    out[n] = calc(avg_g, avg_l)
    for i in range(n + 1, len(closes)):
        avg_g = (avg_g * (n - 1) + gains[i - 1]) / n
        avg_l = (avg_l * (n - 1) + losses[i - 1]) / n
        out[i] = calc(avg_g, avg_l)
    return out


def macd(closes, fast=12, slow=26, sig=9):
    """返回 (dif, dea, hist)。dif=EMA快-EMA慢；dea=dif的EMA；hist=dif-dea。"""
    ef, es = ema(closes, fast), ema(closes, slow)
    dif = [None] * len(closes)
    for i in range(len(closes)):
        if ef[i] is not None and es[i] is not None:
            dif[i] = ef[i] - es[i]
    valid = [x for x in dif if x is not None]
    dea_valid = ema(valid, sig)
    dea = [None] * len(closes)
    j = 0
    for i in range(len(closes)):
        if dif[i] is not None:
            dea[i] = dea_valid[j]
            j += 1
    hist = [None] * len(closes)
    for i in range(len(closes)):
        if dif[i] is not None and dea[i] is not None:
            hist[i] = dif[i] - dea[i]
    return dif, dea, hist


def kdj(highs, lows, closes, n=9, k_s=3, d_s=3):
    """返回 (K, D, J)。RSV 用 n 日高低，K/D 用 1/3 平滑（A股常用参数）。"""
    K = [None] * len(closes)
    D = [None] * len(closes)
    J = [None] * len(closes)
    prev_k, prev_d = 50.0, 50.0
    for i in range(len(closes)):
        if i < n - 1:
            continue
        hh = max(highs[i - n + 1:i + 1])
        ll = min(lows[i - n + 1:i + 1])
        rsv = 0.0 if hh == ll else (closes[i] - ll) / (hh - ll) * 100
        cur_k = (1 / k_s) * rsv + (1 - 1 / k_s) * prev_k
        cur_d = (1 / d_s) * cur_k + (1 - 1 / d_s) * prev_d
        K[i], D[i], J[i] = cur_k, cur_d, 3 * cur_k - 2 * cur_d
        prev_k, prev_d = cur_k, cur_d
    return K, D, J


def bollinger(closes, n=20, mult=2):
    mid = sma(closes, n)
    up = [None] * len(closes)
    lo = [None] * len(closes)
    for i in range(n - 1, len(closes)):
        sd = stddev_pop(closes[i - n + 1:i + 1])
        up[i] = mid[i] + mult * sd
        lo[i] = mid[i] - mult * sd
    return up, mid, lo


def atr(highs, lows, closes, n=14):
    """Wilder ATR。"""
    out = [None] * len(closes)
    tr = [None] * len(closes)
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    if len(closes) <= n:
        return out
    first = sum(tr[1:n + 1]) / n
    out[n] = first
    for i in range(n + 1, len(closes)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def obv(closes, vols):
    out = [0.0] * len(closes)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + vols[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - vols[i]
        else:
            out[i] = out[i - 1]
    return out


def swing_points(highs, lows, w=3):
    """简单摆动高/低点：在 ±w 窗口内为局部极值。返回 (swing_high_idx, swing_low_idx)。"""
    sh, sl = [], []
    for i in range(w, len(highs) - w):
        if highs[i] == max(highs[i - w:i + w + 1]):
            sh.append(i)
        if lows[i] == min(lows[i - w:i + w + 1]):
            sl.append(i)
    return sh, sl


# ============================================================
# 分析主逻辑
# ============================================================

def analyze_ticker(ticker, rng="2y"):
    raw, err = fetch_ohlcv(ticker, rng=rng)
    if err:
        return {"error": err, "ticker": ticker}

    rows = raw["rows"]
    closes = [r["close"] for r in rows]
    highs = [r["high"] for r in rows]
    lows = [r["low"] for r in rows]
    vols = [r["volume"] for r in rows]
    n = len(rows)
    last = n - 1
    price = closes[last]

    # --- 指标 ---
    ma = {p: sma(closes, p) for p in (5, 10, 20, 50, 200)}
    r = rsi(closes)
    dif, dea, hist = macd(closes)
    K, D, J = kdj(highs, lows, closes)
    bu, bm, bl = bollinger(closes)
    a = atr(highs, lows, closes)
    ob = obv(closes, vols)

    def v(series):  # 取末值，保留 None
        return series[last]

    atr_val = v(a)
    atr_pct = (atr_val / price * 100) if atr_val else None

    # 量比
    vol5 = sum(vols[-5:]) / 5 if n >= 5 else vols[last]
    vol20 = sum(vols[-20:]) / 20 if n >= 20 else vols[last]
    vol_ratio = vol5 / vol20 if vol20 else 1.0

    # 高低点 / 突破
    hi60 = max(highs[-61:-1]) if n > 61 else max(highs[:-1])
    lo60 = min(lows[-61:-1]) if n > 61 else min(lows[:-1])
    hi20 = max(highs[-21:-1]) if n > 21 else max(highs[:-1])
    lo20 = min(lows[-21:-1]) if n > 21 else min(lows[:-1])
    is_60d_high = price > hi60
    is_60d_low = price < lo60
    is_20d_high = price > hi20
    pct_from_hi60 = (price - hi60) / hi60 * 100

    # 涨跌幅
    def chg(days):
        if n > days:
            return (price - closes[-days - 1]) / closes[-days - 1] * 100
        return None

    # 支撑/阻力（最近摆动点）
    sh, sl = swing_points(highs, lows, w=3)
    recent_sh = [highs[i] for i in sh if i > n - 120]
    recent_sl = [lows[i] for i in sl if i > n - 120]
    resistance = min([h for h in recent_sh if h > price], default=hi60)
    support = max([l for l in recent_sl if l < price], default=lo60)

    # OBV 趋势（近20日斜率符号）
    obv_trend = "上升" if n > 20 and ob[last] > ob[-21] else ("下降" if n > 20 and ob[last] < ob[-21] else "走平")

    # 背离（近 40 根，比较最后两个摆动高/低点的价 vs RSI）
    divergence = "无明显背离"
    rec_sh = [i for i in sh if i > n - 40]
    rec_sl = [i for i in sl if i > n - 40]
    if len(rec_sh) >= 2 and all(r[i] is not None for i in rec_sh[-2:]):
        p1, p2 = highs[rec_sh[-2]], highs[rec_sh[-1]]
        rr1, rr2 = r[rec_sh[-2]], r[rec_sh[-1]]
        if p2 > p1 and rr2 < rr1:
            divergence = "顶背离（价创新高，RSI 未创新高）— 看空预警"
    if len(rec_sl) >= 2 and all(r[i] is not None for i in rec_sl[-2:]):
        p1, p2 = lows[rec_sl[-2]], lows[rec_sl[-1]]
        rr1, rr2 = r[rec_sl[-2]], r[rec_sl[-1]]
        if p2 < p1 and rr2 > rr1:
            divergence = "底背离（价创新低，RSI 未创新低）— 看多预警"

    # --- 信号旗标（每条标 bull/bear/neutral）---
    flags = []

    def flag(name, value, side, note=""):
        flags.append({"name": name, "value": value, "side": side, "note": note})

    # 均线排列
    mvals = {p: v(ma[p]) for p in (5, 10, 20, 50, 200)}
    if all(mvals[p] is not None for p in (5, 10, 20, 50)):
        if mvals[5] > mvals[10] > mvals[20] > mvals[50]:
            flag("均线排列", "多头排列 (5>10>20>50)", "bull", "趋势向上")
        elif mvals[5] < mvals[10] < mvals[20] < mvals[50]:
            flag("均线排列", "空头排列 (5<10<20<50)", "bear", "趋势向下")
        else:
            flag("均线排列", "缠绕（无明确排列）", "neutral", "震荡或转折")
    if mvals[20]:
        flag("价 vs MA20", f"{'上方' if price > mvals[20] else '下方'} (MA20={mvals[20]:.2f})",
             "bull" if price > mvals[20] else "bear", "中期趋势")
    if mvals[200]:
        flag("价 vs MA200", f"{'上方' if price > mvals[200] else '下方'} (MA200={mvals[200]:.2f})",
             "bull" if price > mvals[200] else "bear", "牛熊分界")

    # RSI
    rv = v(r)
    if rv is not None:
        if rv > 70:
            flag("RSI(14)", f"{rv:.1f} 超买", "bear", "短期或回调，追多需谨慎")
        elif rv < 30:
            flag("RSI(14)", f"{rv:.1f} 超卖", "bull", "短期或反弹")
        else:
            rising = r[last] > r[-3] if n > 3 and r[-3] is not None else None
            flag("RSI(14)", f"{rv:.1f} 中性", "neutral",
                 ("上行" if rising else "下行") if rising is not None else "")

    # MACD
    if v(dif) is not None and v(dea) is not None:
        cross = ""
        if n > 2 and dif[-2] is not None and dea[-2] is not None:
            if dif[-2] <= dea[-2] and dif[last] > dea[last]:
                cross = "金叉"
            elif dif[-2] >= dea[-2] and dif[last] < dea[last]:
                cross = "死叉"
        above_zero = v(dif) > 0
        side = "bull" if v(dif) > v(dea) else "bear"
        flag("MACD", f"DIF={v(dif):.3f} DEA={v(dea):.3f} {cross} ({'零轴上' if above_zero else '零轴下'})",
             side, cross or ("多头" if side == "bull" else "空头"))

    # KDJ
    if v(K) is not None:
        kc = ""
        if n > 2 and K[-2] is not None and D[-2] is not None:
            if K[-2] <= D[-2] and K[last] > D[last]:
                kc = "金叉"
            elif K[-2] >= D[-2] and K[last] < D[last]:
                kc = "死叉"
        kstate = "超买(>80)" if v(K) > 80 else ("超卖(<20)" if v(K) < 20 else "中性")
        kside = "bull" if (v(K) < 20 or kc == "金叉") else ("bear" if (v(K) > 80 or kc == "死叉") else "neutral")
        flag("KDJ", f"K={v(K):.1f} D={v(D):.1f} J={v(J):.1f} {kstate} {kc}", kside)

    # 布林带
    if v(bu) is not None:
        band = v(bu) - v(bl)
        pos = (price - v(bl)) / band if band else 0.5
        if pos > 0.95:
            flag("布林带", f"触/破上轨 (位置 {pos*100:.0f}%)", "bear", "超买区，警惕回落")
        elif pos < 0.05:
            flag("布林带", f"触/破下轨 (位置 {pos*100:.0f}%)", "bull", "超卖区，警惕反弹")
        else:
            flag("布林带", f"带内 (位置 {pos*100:.0f}%, 中轨 {v(bm):.2f})", "neutral")

    # 量价
    if vol_ratio > 1.5:
        vside = "bull" if (chg(1) or 0) > 0 else "bear"
        flag("成交量", f"放量 量比={vol_ratio:.2f}", vside,
             "放量上涨=资金进场" if vside == "bull" else "放量下跌=资金出逃")
    else:
        flag("成交量", f"量能平稳 量比={vol_ratio:.2f}", "neutral")
    flag("OBV", obv_trend, "bull" if obv_trend == "上升" else ("bear" if obv_trend == "下降" else "neutral"),
         "量能累积方向")

    # 突破
    if is_60d_high and vol_ratio > 1.3:
        flag("突破", "放量创60日新高", "bull", "趋势突破，最强多头信号")
    elif is_60d_high:
        flag("突破", "创60日新高（量能不足）", "neutral", "突破需放量确认，谨防假突破")
    elif is_60d_low:
        flag("突破", "创60日新低", "bear", "弱势")

    # 背离
    if "顶背离" in divergence:
        flag("背离", divergence, "bear")
    elif "底背离" in divergence:
        flag("背离", divergence, "bull")

    bull = sum(1 for f in flags if f["side"] == "bull")
    bear = sum(1 for f in flags if f["side"] == "bear")
    net = bull - bear

    # 方向倾向（仅供参考，最终由 skill 综合判断）
    if net >= 3:
        bias = "偏多"
    elif net <= -3:
        bias = "偏空"
    else:
        bias = "观望/震荡"

    # 仓位档（共振数量，对标 stock_screener 的分级）
    confluence = max(bull, bear)
    if confluence >= 5:
        tier = "确信仓"
    elif confluence >= 3:
        tier = "标准仓"
    else:
        tier = "试探仓/观望"

    # ATR 风控建议（按倾向给做多或做空一侧）
    risk = {}
    if atr_val:
        if net >= 0:  # 多头风控
            stop = price - 2 * atr_val
            risk = {
                "side": "做多参考",
                "entry": round(price, 2),
                "stop": round(stop, 2),
                "stop_pct": round((stop - price) / price * 100, 1),
                "target1": round(price + 3 * atr_val, 2),  # 1.5R
                "target2": round(price + 5 * atr_val, 2),  # 2.5R
                "rr_target1": "1.5R", "rr_target2": "2.5R",
                "note": f"止损=入场-2×ATR；支撑位 {support:.2f}，阻力位 {resistance:.2f}",
            }
        else:  # 空头风控
            stop = price + 2 * atr_val
            risk = {
                "side": "做空参考",
                "entry": round(price, 2),
                "stop": round(stop, 2),
                "stop_pct": round((stop - price) / price * 100, 1),
                "target1": round(price - 3 * atr_val, 2),
                "target2": round(price - 5 * atr_val, 2),
                "rr_target1": "1.5R", "rr_target2": "2.5R",
                "note": f"止损=入场+2×ATR；阻力位 {resistance:.2f}，支撑位 {support:.2f}",
            }

    return {
        "ticker": ticker,
        "currency": raw["currency"],
        "exchange": raw["exchange"],
        "date": rows[last]["date"],
        "price": round(price, 2),
        "bars": n,
        "changes": {"1d": chg(1), "5d": chg(5), "20d": chg(20), "60d": chg(60)},
        "atr": round(atr_val, 2) if atr_val else None,
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
        "vol_ratio": round(vol_ratio, 2),
        "high_60d": round(hi60, 2), "low_60d": round(lo60, 2),
        "pct_from_high60": round(pct_from_hi60, 1),
        "support": round(support, 2), "resistance": round(resistance, 2),
        "divergence": divergence,
        "flags": flags,
        "tally": {"bull": bull, "bear": bear, "net": net},
        "bias": bias,
        "position_tier": tier,
        "risk": risk,
    }


# ============================================================
# 回测：在该股自身历史上验证经典信号（兑现"无回测不入卡"）
# ============================================================

def _equity_stats(returns):
    """给一串单笔收益率（小数），算胜率/盈亏比/期望。"""
    if not returns:
        return None
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    payoff = abs(avg_win / avg_loss) if avg_loss else float("inf")
    win_rate = len(wins) / len(returns)
    expectancy = sum(returns) / len(returns)
    return {
        "trades": len(returns),
        "win_rate": round(win_rate * 100, 1),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "payoff": round(payoff, 2) if payoff != float("inf") else "∞",
        "expectancy_pct": round(expectancy * 100, 2),
    }


def backtest_ticker(ticker, rng="5y"):
    raw, err = fetch_ohlcv(ticker, rng=rng)
    if err:
        return {"error": err, "ticker": ticker}
    rows = raw["rows"]
    closes = [r["close"] for r in rows]
    highs = [r["high"] for r in rows]
    lows = [r["low"] for r in rows]
    n = len(rows)

    r = rsi(closes)
    dif, dea, _ = macd(closes)
    ma20 = sma(closes, 20)
    a = atr(highs, lows, closes)

    results = {}

    # 策略1：RSI 均值回归 —— RSI<30 买入，RSI>50 或持有10日后卖出（信号次日开盘≈当日收盘近似）
    rets = []
    i = 30
    while i < n - 1:
        if r[i] is not None and r[i] < 30 and (r[i - 1] is None or r[i - 1] >= 30):
            entry = closes[i]
            j = i + 1
            while j < n and (r[j] is None or r[j] < 50) and j - i < 10:
                j += 1
            j = min(j, n - 1)
            rets.append((closes[j] - entry) / entry)
            i = j + 1
        else:
            i += 1
    results["RSI<30 均值回归"] = _equity_stats(rets)

    # 策略2：MACD 金叉买入，死叉卖出
    rets = []
    pos = None
    for i in range(1, n):
        if None in (dif[i], dea[i], dif[i - 1], dea[i - 1]):
            continue
        gold = dif[i - 1] <= dea[i - 1] and dif[i] > dea[i]
        dead = dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]
        if pos is None and gold:
            pos = closes[i]
        elif pos is not None and dead:
            rets.append((closes[i] - pos) / pos)
            pos = None
    results["MACD 金叉/死叉"] = _equity_stats(rets)

    # 策略3：站上 MA20 买入，跌破 MA20 卖出（趋势跟随）
    rets = []
    pos = None
    for i in range(20, n):
        if ma20[i] is None:
            continue
        above = closes[i] > ma20[i]
        below = closes[i] < ma20[i]
        if pos is None and above and closes[i - 1] <= (ma20[i - 1] or closes[i - 1]):
            pos = closes[i]
        elif pos is not None and below:
            rets.append((closes[i] - pos) / pos)
            pos = None
    results["站上/跌破 MA20"] = _equity_stats(rets)

    # 策略4：20日突破（唐奇安）买入，2×ATR 止损 / 持有20日
    rets = []
    i = 21
    while i < n - 1:
        hi20 = max(highs[i - 20:i])
        if closes[i] > hi20 and a[i]:
            entry = closes[i]
            stop = entry - 2 * a[i]
            j = i + 1
            ret = None
            while j < n and j - i < 20:
                if lows[j] <= stop:
                    ret = (stop - entry) / entry
                    break
                j += 1
            if ret is None:
                j = min(j, n - 1)
                ret = (closes[j] - entry) / entry
            rets.append(ret)
            i = j + 1
        else:
            i += 1
    results["20日突破+2ATR止损"] = _equity_stats(rets)

    return {
        "ticker": ticker,
        "period": f"{rows[0]['date']} ~ {rows[-1]['date']}（{n} 根 K 线）",
        "note": "回测为收盘价近似、单笔满仓、不含手续费/滑点，仅供信号有效性参考；过拟合风险自负。",
        "strategies": results,
    }


# ============================================================
# 输出格式化
# ============================================================

def print_analysis(d):
    if d.get("error"):
        print(f"❌ {d['ticker']}: {d['error']}")
        return
    print(f"\n{'='*60}")
    print(f"  {d['ticker']}  技术面快照  [{d['exchange']} · {d['currency']}]")
    print(f"  截至 {d['date']}  收盘 {d['price']}  ({d['bars']} 根日K)")
    print(f"{'='*60}")
    c = d["changes"]
    print(f"  涨跌幅: 1日 {fmt_pct(c['1d'])} | 5日 {fmt_pct(c['5d'])} | 20日 {fmt_pct(c['20d'])} | 60日 {fmt_pct(c['60d'])}")
    print(f"  ATR(14): {d['atr']} ({d['atr_pct']}%日波动) | 量比: {d['vol_ratio']}")
    print(f"  60日高/低: {d['high_60d']} / {d['low_60d']} (距高点 {d['pct_from_high60']}%)")
    print(f"  支撑/阻力: {d['support']} / {d['resistance']}")
    print(f"  背离: {d['divergence']}")
    print(f"\n  {'信号旗标':<10} {'取值':<40} 方向")
    print(f"  {'-'*58}")
    icon = {"bull": "🟢多", "bear": "🔴空", "neutral": "⚪中"}
    for f in d["flags"]:
        print(f"  {f['name']:<10} {f['value']:<40} {icon[f['side']]}  {f['note']}")
    t = d["tally"]
    print(f"\n  多空计票: 🟢{t['bull']} vs 🔴{t['bear']}  (净 {t['net']:+d})")
    print(f"  方向倾向: {d['bias']}   仓位档: {d['position_tier']}")
    if d["risk"]:
        rk = d["risk"]
        print(f"\n  风控建议 [{rk['side']}]:")
        print(f"    入场 {rk['entry']} | 止损 {rk['stop']} ({rk['stop_pct']}%)")
        print(f"    目标1 {rk['target1']} ({rk['rr_target1']}) | 目标2 {rk['target2']} ({rk['rr_target2']})")
        print(f"    {rk['note']}")
    print(f"\n  ⚠️ 以上为机械计算结果，方向倾向/仓位仅供参考，须由 skill 综合 4 视角判断。")
    print(f"{'='*60}\n")


def print_backtest(d):
    if d.get("error"):
        print(f"❌ {d['ticker']}: {d['error']}")
        return
    print(f"\n{'='*60}")
    print(f"  {d['ticker']}  信号历史回测")
    print(f"  区间: {d['period']}")
    print(f"{'='*60}")
    print(f"  {'策略':<22} {'交易':<5} {'胜率':<7} {'盈亏比':<7} {'期望':<8}")
    print(f"  {'-'*54}")
    for name, s in d["strategies"].items():
        if not s:
            print(f"  {name:<22} 样本不足")
            continue
        print(f"  {name:<22} {s['trades']:<5} {s['win_rate']:<6}% {str(s['payoff']):<7} {s['expectancy_pct']:+}%/笔")
    print(f"\n  注: {d['note']}")
    print(f"{'='*60}\n")


def fmt_pct(x):
    return f"{x:+.1f}%" if x is not None else "n/a"


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="技术面确定性计算引擎")
    sub = p.add_subparsers(dest="cmd")

    pa = sub.add_parser("analyze", help="单标的技术面快照")
    pa.add_argument("ticker")
    pa.add_argument("--range", default="2y", help="数据区间 (默认 2y)")
    pa.add_argument("--json", action="store_true", help="输出原始 JSON")

    pb = sub.add_parser("backtest", help="信号历史回测")
    pb.add_argument("ticker")
    pb.add_argument("--range", default="5y", help="回测区间 (默认 5y)")
    pb.add_argument("--json", action="store_true", help="输出原始 JSON")

    args = p.parse_args()
    if args.cmd == "analyze":
        d = analyze_ticker(args.ticker.upper(), rng=args.range)
        print(json.dumps(d, ensure_ascii=False, indent=2) if args.json else "", end="")
        if not args.json:
            print_analysis(d)
    elif args.cmd == "backtest":
        d = backtest_ticker(args.ticker.upper(), rng=args.range)
        print(json.dumps(d, ensure_ascii=False, indent=2) if args.json else "", end="")
        if not args.json:
            print_backtest(d)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
