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

# Google GenAI SDK
from google import genai
from google.genai import types

MODELS_TO_TRY = ['gemini-3.8-flash', 'gemini-3.1-pro']
DB_PATH = "materials_master.db"

app = FastAPI(title="Interior QS Parametric Studio Hub", version="3.0.0")

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

# API 1: TÌM KIẾM TỔNG HỢP TRONG 9.873 VẬT TƯ
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

# API 2: GỢI Ý THÔNG MINH (SMART RECOMMENDATION CHO TỪNG CẤU KIỆN)
@app.get("/api/materials/recommend")
def recommend_materials(hint: str = Query(..., min_length=2), limit: int = 3):
    conn = get_db()
    cursor = conn.cursor()
    
    clean_hint = hint.replace("+", " ").replace("-", " ").replace("/", " ")
    keywords = [k.strip() for k in clean_hint.split() if len(k.strip()) > 1]
    
    results = []
    # Thử tìm kiếm FTS BM25
    if keywords:
        search_terms = " ".join([f'"{k}"*' for k in keywords[:3]])
        try:
            cursor.execute("""
                SELECT m.ma_vt, m.ten_vt, m.nhom_vt, m.dvt, m.don_gia 
                FROM materials m 
                JOIN materials_fts f ON m.id = f.rowid 
                WHERE materials_fts MATCH ? 
                LIMIT ?
            """, (search_terms, limit))
            results = [dict(r) for r in cursor.fetchall()]
        except Exception:
            results = []
            
    # Dự phòng tìm theo LIKE nếu FTS rỗng
    if not results and keywords:
        like_term = f"%{keywords[0]}%"
        cursor.execute("""
            SELECT ma_vt, ten_vt, nhom_vt, dvt, don_gia 
            FROM materials 
            WHERE ten_vt LIKE ? OR nhom_vt LIKE ?
            LIMIT ?
        """, (like_term, like_term, limit))
        results = [dict(r) for r in cursor.fetchall()]
        
    conn.close()
    return results

# API 3: QUÉT BẢN VẼ TỰ ĐỘNG BẰNG TWO-PASS AI
@app.post("/api/ai/analyze-drawing")
async def analyze_drawing(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="Vui lòng nhập Gemini API Key!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được file ảnh: {str(e)}")

    client = genai.Client(api_key=key)

    # Prompt Two-Pass tổng hợp tối ưu
    prompt = """
    Bạn là một kỹ sư QS (Quantity Surveyor) và chuyên gia bóc tách kết cấu nội thất cao cấp tại Việt Nam.
    Hãy phân tích bản vẽ đính kèm qua 2 góc nhìn (Two-Pass Analysis):
    1. HÌNH HỌC (Geometry Pass):
       - Tên sản phẩm chính (product_name).
       - Kích thước phủ bì tổng thể: width (W, mm), depth (D, mm), height (H, mm).
       - Tách các cấu kiện chính (components): id (C01, C02...), name, shape_type ('extruded_c_shell', 'box', 'cylinder', 'panel'), dimensions {w, d, h}, estimated_qty (khối lượng phôi m2 hoặc bộ).
    2. VẬT LIỆU & SPEC (Spec Pass):
       - Đọc khung ghi chú (GHI CHÚ / SPEC) để trích xuất chủng loại hoàn thiện: da bọc, mút D40, Inox 304 mạ PVD gương/xước, Plywood chịu nước, MDF chống ẩm...
       - Gán material_hint tương ứng cho từng cấu kiện.

    Trả về cú pháp JSON thuần túy (không markdown):
    {
      "product_name": "Tên sản phẩm",
      "overall_dimensions": { "w": 2550, "d": 500, "h": 925 },
      "components": [
        {
          "id": "C01",
          "name": "Khối vỏ C uốn cong",
          "shape_type": "extruded_c_shell",
          "dimensions": { "w": 2100, "d": 500, "h": 925 },
          "material_hint": "Plywood uốn cong + Bọc da mận chín mút D40",
          "unit": "m²",
          "estimated_qty": 4.85
        }
      ]
    }
    """

    last_err = None
    for model_name in MODELS_TO_TRY:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(res.text)
            return {
                "success": True,
                "model_used": model_name,
                "data": data
            }
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(status_code=500, detail=f"Lỗi AI ({MODELS_TO_TRY}): {str(last_err)}")

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
