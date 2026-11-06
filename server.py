from pydantic import BaseModel
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

MODELS_TO_TRY = ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
DB_PATH = "materials_master.db"

app = FastAPI(title="Interior QS Universal 3D Parametric & Multi-Project Hub", version="4.5.0")

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

# Khởi tạo bảng Multi-Project Cache và Quản lý Dự án
def init_project_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        project_name TEXT,
        client_name TEXT,
        location TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Thêm dự án mặc định nếu chưa có
    cursor.execute("""
    INSERT OR IGNORE INTO projects (project_id, project_name, client_name, location)
    VALUES ('DA_VILLA_HN', 'Biệt thự cao cấp Hà Nội', 'Mr. Cường', 'Starlake, Tây Hồ, Hà Nội')
    """)
    cursor.execute("""
    INSERT OR IGNORE INTO projects (project_id, project_name, client_name, location)
    VALUES ('DA_PENTHOUSE_Q2', 'Penthouse Thảo Điền Q2', 'Mrs. Lan', 'An Phú, Thủ Đức, TP.HCM')
    """)
    
    # Bảng cache sản phẩm phân theo project_id
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT,
        image_hash TEXT,
        product_name TEXT,
        product_data TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(project_id, product_name)
    )
    """)
    conn.commit()
    conn.close()

init_project_db()

@app.get("/api/projects")
def get_projects():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT project_id, project_name, client_name, location FROM projects")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

class ProjectCreate(BaseModel):
    project_id: str
    project_name: str
    client_name: Optional[str] = ""
    location: Optional[str] = ""

@app.post("/api/projects/create")
def create_project(p: ProjectCreate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO projects (project_id, project_name, client_name, location)
        VALUES (?, ?, ?, ?)
    """, (p.project_id, p.project_name, p.client_name, p.location))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Đã tạo/cập nhật dự án {p.project_name}"}

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

@app.post("/api/ai/analyze-drawing")
async def analyze_drawing(
    file: UploadFile = File(...),
    project_id: str = Form("DA_VILLA_HN")
):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500, 
            detail="Hệ thống chưa được cấu hình GEMINI_API_KEY trong biến môi trường máy chủ!"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được file ảnh: {str(e)}")

    # Tạo chữ ký đơn giản cho ảnh để kiểm tra cache theo project_id
    img_hash = str(len(contents)) + "_" + file.filename

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT product_data FROM product_cache 
        WHERE project_id = ? AND (image_hash = ? OR product_name = ?)
    """, (project_id, img_hash, file.filename))
    cached = cursor.fetchone()
    if cached:
        conn.close()
        return {
            "success": True,
            "model_used": "cache-memory",
            "data": json.loads(cached["product_data"])
        }

    client = genai.Client(api_key=key)

    prompt = """
    Bạn là Kỹ sư Trưởng QS (Quantity Surveyor) và Chuyên gia Dựng hình 3D Tham số Nội thất.
    Nhiệm vụ: Phân tích bản vẽ kỹ thuật (2D mặt bằng/đứng/cắt hoặc 3D render) của BẤT KỲ sản phẩm nội thất nào (sofa cong, bàn đảo, ghế bành, tủ quần áo, giường ngủ, bàn console, quầy bar, vách ốp...).
    
    Yêu cầu xuất ra cấu trúc JSON thuần túy (không kèm markdown) gồm 3 phần chính để Three.js dựng hình chuẩn xác >95%:
    {
      "product_info": {
        "product_name": "Tên chi tiết sản phẩm",
        "category": "sofa" | "table" | "cabinet" | "chair" | "bed" | "panel" | "other",
        "overall_dimensions": { "w": 4340, "d": 2420, "h": 660 }
      },
      "scene_3d": [
        {
          "name": "Tên cấu kiện 3D",
          "shape": "box" | "cylinder" | "arc_curve" | "c_shell",
          "position": [x_mm, y_mm, z_mm],
          "size": [w_mm, d_mm, h_mm],
          "radius": 0,
          "angle_deg": 90,
          "color": "#c2410c",
          "metalness": 0.1,
          "roughness": 0.5
        }
      ],
      "costsheet_items": [
        {
          "group": "I" | "II" | "III" | "IV" | "V" | "VI" | "VII",
          "code": "Mã vật tư chuẩn xưởng (VD: GTX20.3, MDF9, IN-DEX12, FB LE-01, NC-SOFA, B.BBTN.1...)",
          "name": "Tên cấu kiện và mô tả kỹ thuật",
          "g": 2.45,
          "h": 0.03,
          "i": 0.03,
          "j": 3,
          "unit": "m3" | "m2'" | "Công" | "Bộ",
          "wastage": 1.1 hoặc 1.2,
          "price": 7200000
        }
      ]
    }
    Lưu ý:
    - Nếu có phần khoét lỗ, đâm xuyên âm thì cho j mang giá trị âm (VD: j = -1).
    - Cấu trúc "costsheet_items" phải phân bổ chuẩn theo 7 nhóm chi phí (I: Thầu phụ, II: Vật tư cốt, III: Hoàn thiện, IV: Nhân công, V: Bao bì, VI: Phụ kiện, VII: Source ngoài/Da nỉ).
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
            
            # Lưu vào cache phân theo project_id
            p_name = data.get("product_info", {}).get("product_name", file.filename)
            cursor.execute("""
                INSERT OR REPLACE INTO product_cache (project_id, image_hash, product_name, product_data)
                VALUES (?, ?, ?, ?)
            """, (project_id, img_hash, p_name, json.dumps(data, ensure_ascii=False)))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "model_used": model_name,
                "data": data
            }
        except Exception as e:
            last_err = e
            continue

    conn.close()
    raise HTTPException(status_code=500, detail=f"Lỗi AI ({MODELS_TO_TRY}): {str(last_err)}")

@app.get("/api/dongia/search")
def search_dongia(q: str = Query("", min_length=1), limit: int = 50):
    conn = get_db()
    cursor = conn.cursor()
    query_str = f"%{q.strip()}%"
    cursor.execute("""
        SELECT code, name, spec, unit, wastage, price, supp_code 
        FROM dongia_master 
        WHERE code LIKE ? OR name LIKE ? OR spec LIKE ?
        LIMIT ?
    """, (query_str, query_str, query_str, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/dongia/all")
def get_all_dongia():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT code, name, spec, unit, wastage, price, supp_code FROM dongia_master")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

class DongiaItemUpdate(BaseModel):
    code: str
    name: str
    spec: Optional[str] = ""
    unit: Optional[str] = ""
    wastage: Optional[float] = 1.0
    price: Optional[float] = 0.0
    supp_code: Optional[str] = ""
    supp_info: Optional[str] = ""
    leadtime: Optional[str] = ""
    notes: Optional[str] = ""

@app.post("/api/dongia/update")
def update_dongia_item(item: DongiaItemUpdate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO dongia_master (code, name, spec, unit, wastage, price, supp_code, supp_info, leadtime, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (item.code, item.name, item.spec, item.unit, item.wastage, item.price, item.supp_code, item.supp_info, item.leadtime, item.notes))
    cursor.execute("INSERT INTO dongia_fts(dongia_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Đã cập nhật mã vật tư {item.code}"}

@app.post("/api/dongia/delete")
def delete_dongia_item(code: str = Query(...)):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dongia_master WHERE code = ?", (code,))
    cursor.execute("INSERT INTO dongia_fts(dongia_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Đã xóa mã vật tư {code}"}

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
