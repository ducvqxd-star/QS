import os
import io
import json
import sqlite3
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

# Google GenAI SDK (Mới nhất)
from google import genai
from google.genai import types

# Cấu hình danh sách Model theo quy định
MODELS_TO_TRY = ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
DB_PATH = "materials_master.db"

app = FastAPI(title="Interior QS Parametric Engine API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- API 1: TÌM KIẾM NHANH TRONG GẦN 10.000 VẬT TƯ (TỐC ĐỘ < 5ms) ---
@app.get("/api/materials/search")
def search_materials(q: str = Query("", min_length=1), limit: int = 30):
    conn = get_db()
    cursor = conn.cursor()
    query_str = f"%{q.strip()}%"
    cursor.execute("""
        SELECT ma_vt, ten_vt, nhom_vt, dvt, ncc, don_gia 
        FROM materials 
        WHERE ma_vt LIKE ? OR ten_vt LIKE ? OR nhom_vt LIKE ? OR ncc LIKE ?
        LIMIT ?
    """, (query_str, query_str, query_str, query_str, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- API 2: LẤY CHI TIẾT THEO MÃ VẬT TƯ ---
@app.get("/api/materials/{ma_vt}")
def get_material(ma_vt: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM materials WHERE ma_vt = ?", (ma_vt,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã vật tư")
    return dict(row)

# --- API 3: TẢI BẢN VẼ LÊN VÀ PHÂN TÍCH TỰ ĐỘNG BẰNG AI MULTIMODAL ---
@app.post("/api/ai/analyze-drawing")
async def analyze_drawing(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp Gemini API Key để quét bản vẽ!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {str(e)}")

    client = genai.Client(api_key=key)

    prompt = """
    Bạn là một kỹ sư QS (Quantity Surveyor) và chuyên gia bóc tách kết cấu nội thất cao cấp.
    Nhiệm vụ: Phân tích bản vẽ kỹ thuật 2D (mặt bằng, mặt đứng, mặt cắt) hoặc ảnh render 3D đính kèm.
    
    Hãy trích xuất:
    1. Tên sản phẩm chính (product_name).
    2. Kích thước tổng thể phủ bì: width (dài W, mm), depth (sâu D, mm), height (cao H, mm).
    3. Phân tách sản phẩm thành các khối/cấu kiện hình học độc lập (components):
       - id: mã cấu kiện (VD: C01, C02...)
       - name: tên cấu kiện (VD: Khối vỏ uốn cong, Khối hộp Inox đâm xuyên, Cột vách kịch trần...)
       - shape_type: 'box', 'cylinder', hoặc 'extruded_c_shell'
       - dimensions: { w, d, h }
       - material_hint: gợi ý tên vật liệu (VD: da, mút, inox 304, gỗ sồi, plywood...)
       - unit: 'm2', 'md', 'bộ', 'cái'
       - estimated_qty: khối lượng diện tích phôi ước tính (float)
    
    BẮT BUỘC trả về cú pháp JSON thuần túy (không kèm markdown ```json):
    {
      "product_name": "Tên sản phẩm",
      "overall_dimensions": { "w": 2550, "d": 500, "h": 925 },
      "components": [
        {
          "id": "C01",
          "name": "Khối vỏ C uốn cong",
          "shape_type": "extruded_c_shell",
          "dimensions": { "w": 2100, "d": 500, "h": 925 },
          "material_hint": "Plywood uốn cong + Bọc da/vải cao cấp",
          "unit": "m2",
          "estimated_qty": 4.85
        }
      ]
    }
    """

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            return {
                "success": True,
                "model_used": model_name,
                "data": data
            }
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=500, detail=f"Không thể phân tích qua các model {MODELS_TO_TRY}. Lỗi: {str(last_error)}")

# Mount static files phục vụ giao diện Web lên trên root
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print("  MAY CHU HE THONG DA SAN SANG!")
    print("  Truy cap ung dung tai: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
