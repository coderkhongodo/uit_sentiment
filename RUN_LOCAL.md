# 🚀 Hướng dẫn chạy ứng dụng Local

## 📋 **Yêu cầu hệ thống**
- Python 3.8+
- Node.js 16+
- 4GB RAM (để load PhoBERT model)

## ⚡ **Cách chạy nhanh**

### **Bước 1: Chạy Backend**
```bash
# Mở terminal đầu tiên
cd backend
pip install -r requirements_minimal.txt
python run.py
```

Backend sẽ chạy tại: **http://localhost:8000**

### **Bước 2: Chạy Frontend** 
```bash
# Mở terminal thứ hai
cd frontend
npm install
npm start
```

Frontend sẽ chạy tại: **http://localhost:3000**

## 🎯 **Sử dụng ứng dụng**

1. **Truy cập**: http://localhost:3000
2. **Nhập văn bản tiếng Việt** để phân tích
3. **Xem kết quả** với độ tin cậy và xác suất
4. **Khám phá Dashboard** để xem thống kê

## 📊 **Các trang chính**

- **Trang chủ** (`/`) - Phân tích cảm xúc
- **Dashboard** (`/dashboard`) - Biểu đồ và thống kê  
- **Analytics** (`/analytics`) - Lịch sử chi tiết
- **Admin** (`/admin`) - Quản trị hệ thống

## 🔧 **API Endpoints**

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Analyze**: POST http://localhost:8000/analyze
- **Analytics**: GET http://localhost:8000/analytics

## ❓ **Troubleshooting**

### **Lỗi cài đặt Python packages:**
```bash
# Thử cài từng package
pip install fastapi uvicorn pydantic python-multipart aiosqlite
pip install torch transformers numpy pandas scikit-learn
```

### **Lỗi cài đặt Node packages:**
```bash
# Xóa node_modules và cài lại
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### **Model không load được:**
- Kiểm tra thư mục `saved_results/final_model` có tồn tại
- Đảm bảo có đủ RAM (4GB+)

## 🎮 **Văn bản mẫu để test**

```
Tích cực: "Thầy giảng rất hay và nhiệt tình, tôi rất thích môn học này"
Tiêu cực: "Môn học này khó quá, tôi không hiểu gì cả"  
Trung lập: "Hôm nay trời đẹp, tôi đi học bình thường"
```

---

**Lưu ý**: Đảm bảo chạy backend trước, sau đó mới chạy frontend!