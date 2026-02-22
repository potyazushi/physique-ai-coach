# app/coach_engine.py
# CHANGE: 「直近7件」ではなく「直近7日（created_atの日付単位）」の時系列に正規化
# CHANGE: 時系列は必ず昇順（古い→新しい）で計算
# CHANGE: trend は線形回帰（kg/day）で推定し、週あたり±1.5%でクリップ
# CHANGE: 予測体重が暴走しないよう、現実的な範囲にクリップ
# CHANGE: データ不足時は計算を安全側にフォールバック

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DailyPoint:
    day: date
    weight: float
    body_fat: Optional[float] = None
    calorie: Optional[float] = None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def normalize_to_daily_points(
    records: List[dict],
    max_days: int = 60,
    keep: str = "last",
) -> List[DailyPoint]:
    """
    records: [{"created_at": datetime, "weight": float, "body_fat": float|None, "calorie": float|None}, ...]
    同日に複数レコードがある場合、keep="last" ならその日の最後の1件を採用（多重送信対策）。
    """
    if not records:
        return []

    # created_at を確実に datetime として扱い、昇順に
    cleaned = []
    for r in records:
        ca = r.get("created_at")
        if isinstance(ca, str):
            # ISO想定（念のため）
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                continue
        if not isinstance(ca, datetime):
            continue

        w = _safe_float(r.get("weight"))
        if w is None:
            continue
        bf = _safe_float(r.get("body_fat"))
        cal = _safe_float(r.get("calorie"))
        cleaned.append((ca, w, bf, cal))

    cleaned.sort(key=lambda x: x[0])  # old -> new

    # 直近 max_days に絞る（created_at 기준）
    cutoff = datetime.utcnow().date() - timedelta(days=max_days - 1)

    # 日次集約
    by_day = {}
    for ca, w, bf, cal in cleaned:
        d = ca.date()
        if d < cutoff:
            continue
        if d not in by_day:
            by_day[d] = []
        by_day[d].append((ca, w, bf, cal))

    points: List[DailyPoint] = []
    for d in sorted(by_day.keys()):
        rows = by_day[d]
        rows.sort(key=lambda x: x[0])  # within day
        if keep == "first":
            ca, w, bf, cal = rows[0]
        elif keep == "avg":
            w = float(np.mean([x[1] for x in rows]))
            bf_vals = [x[2] for x in rows if x[2] is not None]
            cal_vals = [x[3] for x in rows if x[3] is not None]
            bf = float(np.mean(bf_vals)) if bf_vals else None
            cal = float(np.mean(cal_vals)) if cal_vals else None
        else:
            ca, w, bf, cal = rows[-1]  # last
        points.append(DailyPoint(day=d, weight=w, body_fat=bf, calorie=cal))

    return points


def weekly_drop_pct(daily: List[DailyPoint], window_days: int = 7) -> float:
    """
    直近 window_days 日の「開始（古い）→終了（新しい）」で週減少率を計算。
    """
    if len(daily) < 2:
        return 0.0

    # 直近window_days日だけ切り出し
    last = daily[-window_days:]
    if len(last) < 2:
        return 0.0

    start = last[0].weight
    end = last[-1].weight
    if start <= 0:
        return 0.0
    return (start - end) / start * 100.0


def estimate_trend_kg_per_day(daily: List[DailyPoint], window_days: int = 14) -> float:
    """
    線形回帰で trend(kg/day) を推定（負なら減量、正なら増量）。
    """
    if len(daily) < 2:
        return 0.0

    last = daily[-window_days:]
    if len(last) < 2:
        return 0.0

    # x: 日数(0..n-1) ではなく実日付差を使う（欠測があっても破綻しにくい）
    d0 = last[0].day
    xs = np.array([(p.day - d0).days for p in last], dtype=float)
    ys = np.array([p.weight for p in last], dtype=float)

    if np.all(xs == xs[0]):
        return 0.0

    # y = a*x + b
    a, b = np.polyfit(xs, ys, 1)
    trend = float(a)  # kg/day
    if np.isnan(trend) or np.isinf(trend):
        return 0.0
    return trend


def clip_prediction(current: float, predicted: float, weeks_out: int, weekly_cap_pct: float = 1.5) -> float:
    """
    予測体重が暴走しないように、週あたり±weekly_cap_pct% の範囲にクリップ。
    """
    if weeks_out <= 0:
        return current

    cap = current * (weekly_cap_pct / 100.0) * weeks_out
    lo = current - cap
    hi = current + cap
    return float(np.clip(predicted, lo, hi))


def calculate_cut_score(
    daily: List[DailyPoint],
    weeks_out: int,
    ideal_weekly_drop_pct: float = 0.7,
) -> Tuple[int, dict]:
    """
    0-100点のスコアと、表示用メトリクスを返す。
    """
    metrics = {
        "weekly_drop_pct": 0.0,
        "trend_kg_per_day": 0.0,
        "predicted_weight": None,
        "stability_std": None,
        "plateau": False,
    }

    if not daily:
        return 0, metrics

    current = daily[-1].weight
    wdrop = weekly_drop_pct(daily, window_days=7)
    trend = estimate_trend_kg_per_day(daily, window_days=14)

    # 予測： current + trend * (weeks_out*7)
    raw_pred = current + trend * float(max(weeks_out, 0) * 7)
    pred = clip_prediction(current=current, predicted=raw_pred, weeks_out=max(weeks_out, 0), weekly_cap_pct=1.5)

    # 安全クリップ（体重としてありえない値を出さない）
    pred = float(np.clip(pred, 30.0, 200.0))

    # 安定性（直近7日標準偏差）
    last7 = daily[-7:]
    std = float(np.std([p.weight for p in last7], ddof=0)) if len(last7) >= 2 else 0.0

    # 停滞判定（直近3日で変化が小さい）
    plateau = False
    if len(daily) >= 3:
        w3 = [p.weight for p in daily[-3:]]
        plateau = (max(w3) - min(w3)) < 0.2

    metrics.update(
        {
            "weekly_drop_pct": float(wdrop),
            "trend_kg_per_day": float(trend),
            "predicted_weight": float(pred),
            "stability_std": float(std),
            "plateau": bool(plateau),
        }
    )

    # ---- scoring（0-100）----
    # pace: ideal 0.7% に近いほど高い（0%でも悪すぎないが低め）
    # 過大（>2%）や増量（負）をペナルティ
    pace_diff = abs(wdrop - ideal_weekly_drop_pct)
    pace_score = max(0.0, 40.0 - pace_diff * 20.0)  # 0.7±1.0で20点程度

    if wdrop < -0.2:  # 週で増えてる（悪化）
        pace_score *= 0.6
    if wdrop > 2.0:   # 落としすぎ
        pace_score *= 0.7

    # stability: std 0.2以下なら満点、0.8以上で低い
    stability_score = float(np.clip(30.0 * (1.0 - (std / 0.8)), 0.0, 30.0))

    # deadline: weeks_out が長いほど影響小、近いほど厳密（ここでは簡易）
    # 予測が「現在から過大に増える」場合に減点
    deadline_score = 30.0
    if pred > current + 1.0:
        deadline_score -= min(20.0, (pred - current) * 5.0)

    total = pace_score + stability_score + deadline_score
    if plateau:
        total -= 10.0

    total = float(np.clip(total, 0.0, 100.0))
    return int(round(total)), metrics


def plot_weight_chart(daily: List[DailyPoint], output_path: str) -> None:
    """
    static/weight_chart.png を生成
    """
    if not daily:
        return

    xs = [p.day.strftime("%m/%d") for p in daily[-30:]]
    ys = [p.weight for p in daily[-30:]]

    plt.figure(figsize=(8, 3))
    plt.plot(xs, ys)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()