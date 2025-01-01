from fastapi import FastAPI
from .database import DatabaseConnection

app = FastAPI(title="Startup Wellness API")

# データベース接続
db = DatabaseConnection()

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    await db.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    await db.cleanup()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Welcome to Startup Wellness API"}