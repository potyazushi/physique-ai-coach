# app/models.py
# CHANGE: body_fat / calorie を nullable=True として安定運用（過去データ互換）
# CHANGE: table名を固定（weight_records）

from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime

from .database import Base


class WeightRecord(Base):
    __tablename__ = "weight_records"

    id = Column(Integer, primary_key=True, index=True)
    weight = Column(Float, nullable=False)
    body_fat = Column(Float, nullable=True)   # 過去データ/未入力対応
    calorie = Column(Float, nullable=True)    # UI入力があるので保持（任意）
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class NutritionRecord(Base):
    __tablename__ = "nutrition_records"

    id = Column(Integer, primary_key=True, index=True)
    protein = Column(Float, nullable=True)
    carb = Column(Float, nullable=True)
    fat = Column(Float, nullable=True)
    calorie = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)