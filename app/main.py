# app/main.py
# CHANGE: startupで init_db() 実行（create_all + schema整合）
# CHANGE: static mount を明示

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .database import init_db

app = FastAPI(title="Physique AI Coach")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)


@app.on_event("startup")
def on_startup():
    init_db()