# app/api.py
# CHANGE: GET "/" はトップページ(home.html)を返す
# CHANGE: GET "/coach" は従来のコーチ画面(index.html)を返す
# NOTE: API(/api/...)はそのまま。フロントJSのfetchもそのまま動く

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, date, timedelta, time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import WeightRecord
from .coach_engine import (
    normalize_to_daily_points,
    calculate_cut_score,
    plot_weight_chart,
)

load_dotenv()

router = APIRouter()
templates = Jinja2Templates(directory="templates")

STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
CHART_PATH = STATIC_DIR / "weight_chart.png"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CONTEST_DATE = os.getenv("CONTEST_DATE", "")
DEBUG_LOGIC = os.getenv("DEBUG_LOGIC", "0") == "1"

# JSTで「同日判定」するための固定オフセット（まずは+9でOK）
JST_OFFSET_HOURS = 9


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def parse_contest_date() -> Optional[date]:
    if not CONTEST_DATE:
        return None
    try:
        return datetime.strptime(CONTEST_DATE, "%Y-%m-%d").date()
    except Exception:
        return None


def calc_weeks_out(contest: Optional[date]) -> int:
    if contest is None:
        return 0
    today = datetime.utcnow().date()
    days = (contest - today).days
    if days <= 0:
        return 0
    return int((days + 6) // 7)  # 切り上げ


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
        else:
            t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


def validate_inputs(weight: float, body_fat: Optional[float], calorie: Optional[float]) -> None:
    if not (30.0 <= weight <= 200.0):
        raise HTTPException(status_code=400, detail="weight は 30〜200kg の範囲で入力してください。")
    if body_fat is not None and not (3.0 <= body_fat <= 60.0):
        raise HTTPException(status_code=400, detail="body_fat は 3〜60% の範囲で入力してください。")
    if calorie is not None and not (800.0 <= calorie <= 6000.0):
        raise HTTPException(status_code=400, detail="calorie は 800〜6000kcal の範囲で入力してください。")


def utc_range_for_jst_day(target_jst_day: date) -> tuple[datetime, datetime]:
    """
    JSTの1日（00:00〜24:00）を、DB保存しているUTC(naive)範囲に変換する。
    DB created_at は datetime.utcnow() の naive UTC 前提。
    """
    start_jst = datetime.combine(target_jst_day, time.min)
    start_utc = start_jst - timedelta(hours=JST_OFFSET_HOURS)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc


def jst_today() -> date:
    return (datetime.utcnow() + timedelta(hours=JST_OFFSET_HOURS)).date()


def find_today_record(db: Session) -> Optional[WeightRecord]:
    """
    JSTの今日に該当するレコード（その日1件運用）を探す。
    複数存在してしまった場合は最新(created_at desc)を返す。
    """
    today = jst_today()
    start_utc, end_utc = utc_range_for_jst_day(today)

    return (
        db.query(WeightRecord)
        .filter(WeightRecord.created_at >= start_utc)
        .filter(WeightRecord.created_at < end_utc)
        .order_by(WeightRecord.created_at.desc())
        .first()
    )


async def analyze_with_openai_vision(image_path: Path, weight: float, body_fat: Optional[float], calorie: Optional[float], weeks_out: int) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "observations": ["（OPENAI_API_KEY未設定のため画像分析をスキップ）"],
            "actions": ["まずは数値と記録運用を安定させましょう。"],
            "uncertainty_note": "画像分析なし",
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        b = image_path.read_bytes()
        b64 = base64.b64encode(b).decode("utf-8")

        user_context = {
            "weight": weight,
            "body_fat": body_fat,
            "calorie": calorie,
            "weeks_out": weeks_out,
        }

        prompt = f"""
あなたはフィジーク大会向けの減量コーチです。
画像（体型）と当日の数値から、短く具体的にアドバイスしてください。

制約:
- 必ずJSONのみで返す（コードフェンス禁止）
- キーは observations/actions/uncertainty_note
- observations: 最大5個
- actions: 最大5個（今週の優先順位つき）
- uncertainty_note: 不確実性があれば1文

当日の数値: {json.dumps(user_context, ensure_ascii=False)}
"""

        resp = client.responses.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                    ],
                }
            ],
        )

        text = _strip_code_fence(resp.output_text)
        data = _safe_json_loads(text)
        if isinstance(data, dict):
            return {
                "observations": data.get("observations", []),
                "actions": data.get("actions", []),
                "uncertainty_note": data.get("uncertainty_note", ""),
            }

        return {
            "observations": ["画像分析のJSON解析に失敗しました（フォールバック）"],
            "actions": ["数値ベースで今週の減量ペースを整えましょう。"],
            "uncertainty_note": "モデル出力がJSONではありませんでした。",
        }

    except Exception as e:
        return {
            "observations": [f"画像分析に失敗: {type(e).__name__}"],
            "actions": ["画像なしで数値ベースの調整を続けましょう。"],
            "uncertainty_note": "Vision呼び出し失敗",
        }


# -------------------------
# Pages
# -------------------------

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    # CHANGE: トップページ（LP）
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/coach", response_class=HTMLResponse)
def coach_page(request: Request):
    # CHANGE: 従来の画面は /coach に移動
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------
# APIs
# -------------------------

@router.post("/api/coach")
async def coach(
    weight: float = Form(...),
    body_fat: Optional[float] = Form(None),
    calorie: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    validate_inputs(weight, body_fat, calorie)

    contest = parse_contest_date()
    weeks_out = calc_weeks_out(contest)

    # 1) 画像保存（任意）
    image_path: Optional[Path] = None
    if image is not None and image.filename:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        suffix = Path(image.filename).suffix.lower() or ".jpg"
        image_path = UPLOAD_DIR / f"{ts}{suffix}"
        image_bytes = await image.read()
        image_path.write_bytes(image_bytes)

    # 2) JST同日レコードがあればUPDATE、なければINSERT
    today_rec = find_today_record(db)
    now = datetime.utcnow()

    if today_rec is None:
        rec = WeightRecord(
            weight=float(weight),
            body_fat=float(body_fat) if body_fat is not None else None,
            calorie=float(calorie) if calorie is not None else None,
            created_at=now,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        upsert_mode = "insert"
        saved_id = rec.id
    else:
        today_rec.weight = float(weight)
        today_rec.body_fat = float(body_fat) if body_fat is not None else None
        today_rec.calorie = float(calorie) if calorie is not None else None
        today_rec.created_at = now
        db.commit()
        upsert_mode = "update"
        saved_id = today_rec.id

    # 3) 直近データ取得（最大60日）→ 日次正規化
    rows = db.query(WeightRecord).order_by(WeightRecord.created_at.asc()).all()
    records = [
        {
            "created_at": r.created_at,
            "weight": r.weight,
            "body_fat": r.body_fat,
            "calorie": getattr(r, "calorie", None),
        }
        for r in rows
    ]
    daily = normalize_to_daily_points(records, max_days=60, keep="last")

    # 4) ロジック計算
    score, metrics = calculate_cut_score(daily, weeks_out=weeks_out, ideal_weekly_drop_pct=0.7)

    # 5) チャート生成
    try:
        plot_weight_chart(daily, str(CHART_PATH))
    except Exception:
        pass

    # 6) Vision/LLM（任意）
    ai = {"observations": [], "actions": [], "uncertainty_note": ""}
    if image_path is not None:
        ai = await analyze_with_openai_vision(
            image_path,
            weight=float(weight),
            body_fat=body_fat,
            calorie=calorie,
            weeks_out=weeks_out,
        )

    if DEBUG_LOGIC:
        print("=== DEBUG_LOGIC ===")
        print("upsert_mode:", upsert_mode, "saved_id:", saved_id)
        print("weeks_out:", weeks_out, "contest:", contest)
        print("daily_points(len):", len(daily))
        print("daily last 10:")
        for p in daily[-10:]:
            print("  ", p.day.isoformat(), p.weight)
        print("metrics:", metrics)
        print("score:", score)
        print("===================")

    # 7) UI返却（最終ガード）
    weekly = float(metrics.get("weekly_drop_pct") or 0.0)
    predicted = metrics.get("predicted_weight")
    if predicted is None:
        predicted = float(weight)

    weekly = float(max(min(weekly, 5.0), -5.0))
    predicted = float(max(min(float(predicted), 200.0), 30.0))

    return JSONResponse(
        {
            "cut_score": score,
            "weeks_out": weeks_out,
            "weekly_drop_pct": round(weekly, 2),
            "predicted_weight": round(predicted, 1),
            "chart_url": f"/static/weight_chart.png?ts={int(datetime.utcnow().timestamp())}",
            "ai": {
                "observations": ai.get("observations", []),
                "actions": ai.get("actions", []),
                "uncertainty_note": ai.get("uncertainty_note", ""),
            },
            "save": {"mode": upsert_mode, "record_id": saved_id, "jst_day": str(jst_today())},
        }
    )


@router.get("/api/records")
def records(db: Session = Depends(get_db)):
    rows = (
        db.query(WeightRecord)
        .order_by(WeightRecord.created_at.desc())
        .limit(30)
        .all()
    )
    return [
        {
            "id": r.id,
            "weight": r.weight,
            "body_fat": r.body_fat,
            "calorie": getattr(r, "calorie", None),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.put("/api/records/{record_id}")
def update_record(
    record_id: int,
    weight: float = Form(...),
    body_fat: Optional[float] = Form(None),
    calorie: Optional[float] = Form(None),
    db: Session = Depends(get_db),
):
    validate_inputs(weight, body_fat, calorie)

    rec = db.query(WeightRecord).filter(WeightRecord.id == record_id).first()
    if rec is None:
        raise HTTPException(status_code=404, detail="record not found")

    rec.weight = float(weight)
    rec.body_fat = float(body_fat) if body_fat is not None else None
    rec.calorie = float(calorie) if calorie is not None else None
    db.commit()

    return {"ok": True, "id": rec.id}