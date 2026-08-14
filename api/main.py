"""
FastAPI バックエンド
フロントエンドからのリクエストを受けてML予想を返す
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import logging
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import DB_PATH, model_path
from predict import predict_race, build_race_id, RACECOURSE_NAME_TO_CODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="競馬AI予想API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

META_PATH = model_path("model_meta.json")


class PredictRequest(BaseModel):
    date: str           # "2026-08-10"
    racecourse: str     # "阪神"
    race_num: int       # 11
    kai: int = 1        # 開催回数
    day: int = 1        # 開催日数


@app.get("/")
def root():
    return {"status": "ok", "message": "競馬AI予想API"}


@app.get("/racecourses")
def get_racecourses():
    return {"racecourses": list(RACECOURSE_NAME_TO_CODE.keys())}


@app.get("/model/info")
def model_info():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    raise HTTPException(status_code=404, detail="モデルが未学習です。先に train.py を実行してください。")


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        race_id = build_race_id(req.date, req.racecourse, req.kai, req.day, req.race_num)
        logger.info(f"予想リクエスト: race_id={race_id}")

        result = predict_race(race_id)

        return {
            "status": "success",
            "race_id": race_id,
            "race_info": {
                "date": req.date,
                "racecourse": req.racecourse,
                "race_num": req.race_num,
                "distance": result["race_info"].get("distance"),
                "surface": result["race_info"].get("surface"),
                "track_condition": result["race_info"].get("track_condition"),
                "weather": result["race_info"].get("weather"),
            },
            "horses": result["horses"],
            "recommendation": result["recommendation"],
            "predicted_at": datetime.now().isoformat(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"予想エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"予想処理中にエラーが発生しました: {str(e)}")


@app.post("/result/register")
def register_result(body: dict):
    """
    レース結果を登録してモデルを自動再学習する
    body: { race_id, results: [{horse_num, finish_pos}] }
    """
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    race_id = body.get("race_id")
    for r in body.get("results", []):
        c.execute(
            "UPDATE race_results SET finish_pos=? WHERE race_id=? AND horse_num=?",
            (r["finish_pos"], race_id, r["horse_num"])
        )
    conn.commit()
    conn.close()

    # 非同期で再学習をトリガー（本番ではCelery等を使う）
    import threading
    def retrain():
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
            from train import retrain_with_new_data
            retrain_with_new_data()
            logger.info("モデル再学習完了")
        except Exception as e:
            logger.error(f"再学習エラー: {e}")

    t = threading.Thread(target=retrain, daemon=True)
    t.start()

    return {"status": "accepted", "message": "結果を登録しました。バックグラウンドで再学習を開始します。"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
