import os
import io
import json
import sqlite3
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Google GenAI SDK
from google import genai
from google.genai import types

# Cấu hình chuỗi mô hình ưu tiên
MODELS_TO_TRY = ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
DB_PATH = "materials_master.db"

app = FastAPI(title="Interior QS Universal 3D Parametric Hub", version="4.0.0")

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

@app.get("/api/materials/recommend")
def recommend_materials(hint: str = Query(..., min_length=2), limit: int = 3):
    conn = get_db()
    cursor = conn.cursor()
    clean_hint = hint.replace("+", " ").replace("-", " ").replace("/", " ")
    keywords = [k.strip() for k in clean_hint.split() if len(k.strip()) > 1]
    results = []
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

    prompt = """
    Bạn là Kỹ sư Trưởng QS và Chuyên gia Dựng hình CAD 3D Nội thất.
    Nhiệm vụ: Phân tích kỹ bản vẽ kỹ thuật 2D (mặt bằng, mặt đứng, mặt bên) và ảnh phối cảnh 3D của sản phẩm nội thất đính kèm (bàn trà lồng, bàn đảo, sofa cong, tủ, giường, vách...).
    
    Yêu cầu xuất ra cấu trúc JSON thuần túy (không kèm markdown):
    {
      "product_info": {
        "product_name": "Tên chuẩn xác của sản phẩm (VD: Bàn Trà Không Gian SP09, Ghế Sofa Góc Cong...)",
        "category": "table" | "sofa" | "cabinet" | "chair" | "bed" | "other",
        "overall_dimensions": { "w": 1535, "d": 1270, "h": 370 }
      },
      "scene_3d": [
        {
          "name": "Tên khối cấu tạo",
          "shape": "cylinder" | "oval" | "box" | "arc_curve",
          "position": [x_mm, y_mm, z_mm],
          "size": [w_mm, d_mm, h_mm],
          "radius": 585,
          "height": 85,
          "rotation_deg": 30,
          "color": "#334155",
          "roughness": 0.5,
          "metalness": 0.2
        }
      ],
      "costsheet_items": [
        {
          "group": "I" | "II" | "III" | "IV" | "V" | "VI" | "VII",
          "code": "Mã vật tư chuẩn (VD: MDF9, GTX20.3, IN-DEX12, FB LE-01, NC-TU, B.BBTN.1...)",
          "name": "Tên chi tiết cấu kiện",
          "g": 1.17,
          "h": 1.17,
          "i": 0.085,
          "j": 1,
          "unit": "m2'" | "m3" | "Công" | "Bộ",
          "wastage": 1.15,
          "price": 1250000
        }
      ]
    }
    Quy tắc bắt buộc:
    - Nếu là bàn mặt tròn -> shape: "cylinder" (với radius = đường kính / 2, height = độ dày mặt).
    - Nếu là mặt bàn hình elip / hạt đậu hữu cơ -> shape: "oval" (với size: [chiều dài, chiều rộng, độ dày]).
    - Cấu trúc "costsheet_items" phải phân bổ đúng 7 nhóm chi phí (I đến VII), nếu khoét lỗ/giảm trừ thì j mang giá trị âm (j = -1).
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
