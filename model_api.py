import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Khởi tạo app FastAPI
app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# 2. Tải mô hình TỐT NHẤT của bạn (lgbm_model.pkl)
# File này phải nằm cùng thư mục với model_api.py
try:
    model = joblib.load('lgbm_model.pkl')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'lgbm_model.pkl'.")
    print("Hãy đảm bảo file mô hình nằm cùng thư mục với file API này.")
    exit()

# 3. Định nghĩa các cột MÔ HÌNH (Rất quan trọng)
# Đây là danh sách các cột MÀ MÔ HÌNH ĐÃ ĐƯỢC TRAIN
# Tên và thứ tự phải khớp 100%
MODEL_COLUMNS = [
    'age', 'ap_hi', 'ap_lo', 'bmi', 'smoke', 'alco', 'active',
    'gender_2', 'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'
]

# 4. Định nghĩa Input Schema (Pydantic)
# Đây là cấu trúc dữ liệu mà Node.js (req.body) SẼ GỬI LÊN
# FastAPI sẽ tự động kiểm tra xem req.body có đúng định dạng này không
class PatientData(BaseModel):
    age: int
    gender: int      # 1 = Nữ, 2 = Nam
    height: float    # Tính bằng cm (ví dụ: 170)
    weight: float    # Tính bằng kg (ví dụ: 70)
    ap_hi: int
    ap_lo: int
    cholesterol: int # 1, 2, hoặc 3
    gluc: int        # 1, 2, hoặc 3
    smoke: int       # 0 hoặc 1
    alco: int        # 0 hoặc 1
    active: int      # 0 hoặc 1
    # Bạn có thể thêm các trường khác mà web app gửi lên (như patientName)
    # nhưng chúng sẽ không được dùng trong mô hình
    patientName: str | None = None 

# 5. Tạo Endpoint /predict
@app.post("/predict")
async def predict_heart_disease(data: PatientData):
    """
    Nhận dữ liệu bệnh nhân và trả về xác suất mắc bệnh tim.
    """

    # 1. Tiền xử lý (Giống hệt dashboard đã sửa lỗi)
    # Tính BMI
    if data.height <= 0:
        return {"error": "Chiều cao không thể bằng 0"}

    bmi = data.weight / ((data.height / 100) ** 2)

    # 2. Tạo DataFrame rỗng với đúng các cột
    input_df = pd.DataFrame(columns=MODEL_COLUMNS, index=[0])
    input_df.fillna(0, inplace=True) # Điền 0 vào tất cả

    # 3. Điền dữ liệu từ 'data' (Pydantic model)
    input_df['age'] = data.age
    input_df['ap_hi'] = data.ap_hi
    input_df['ap_lo'] = data.ap_lo
    input_df['bmi'] = bmi
    input_df['smoke'] = data.smoke
    input_df['alco'] = data.alco
    input_df['active'] = data.active

    # 4. One-Hot Encoding (giống hệt lúc train)
    if data.gender == 2:
        input_df['gender_2'] = 1
    if data.cholesterol == 2:
        input_df['cholesterol_2'] = 1
    elif data.cholesterol == 3:
        input_df['cholesterol_3'] = 1
    if data.gluc == 2:
        input_df['gluc_2'] = 1
    elif data.gluc == 3:
        input_df['gluc_3'] = 1

    # 5. Dự đoán
    # Đảm bảo thứ tự cột chính xác
    input_df = input_df[MODEL_COLUMNS] 

    # Lấy xác suất của class 1 (bị bệnh)
    probability = model.predict_proba(input_df)[0][1] 

    # 6. Trả về kết quả JSON (Node.js sẽ nhận cái này)
    return {"probability": probability}

# (Tùy chọn) Endpoint gốc để kiểm tra API có "sống" không
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running."}