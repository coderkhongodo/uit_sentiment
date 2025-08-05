# 🇻🇳 Vietnamese Sentiment Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Vietnamese sentiment analysis system featuring a **PhoBERT-based ML model** with **94%+ accuracy** and a modern **full-stack web application** for real-time sentiment analysis.

## ✨ Key Features

### 🎯 **Machine Learning**
- **High Accuracy**: 94%+ test accuracy on UiT-VSFC dataset
- **PhoBERT-based**: State-of-the-art Vietnamese language model
- **Production Ready**: Optimized for real-world deployment
- **Model Flexibility**: Easy to swap and integrate different models

### 🌐 **Web Application**
- **Real-time Analysis**: Instant sentiment prediction with confidence scores
- **Interactive Dashboard**: Comprehensive analytics and visualizations
- **Admin Panel**: System monitoring and management tools
- **Responsive Design**: Modern UI with Tailwind CSS
- **RESTful API**: Well-documented FastAPI backend

### 📊 **Analytics & Monitoring**
- **Detailed Metrics**: Precision, recall, F1-score tracking
- **Usage Statistics**: Analysis history and trends
- **Performance Monitoring**: Real-time system health checks
- **Data Visualization**: Interactive charts and graphs

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **4GB+ RAM** (for PhoBERT model)
- **Git**

### 🔧 Installation & Setup

#### 1. Clone Repository
```bash
git clone https://github.com/coderkhongodo/uit_sentiment.git
cd uit_SentimentAnalysis
```

#### 2. Backend Setup
```bash
cd backend
pip install -r requirements_minimal.txt
python run.py
```
**Backend URL**: http://localhost:8000

#### 3. Frontend Setup
```bash
# Open new terminal
cd frontend
npm install
npm start
```
**Frontend URL**: http://localhost:3000

### 🎯 Usage

#### Web Interface
1. **Access**: http://localhost:3000
2. **Enter Vietnamese text** for sentiment analysis
3. **View results** with confidence scores and probabilities
4. **Explore Dashboard** for analytics and insights

#### API Usage
```python
import requests

# Analyze sentiment via API
response = requests.post("http://localhost:8000/analyze", 
    json={"text": "Thầy giảng rất hay và nhiệt tình"})
result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Programmatic Usage
```python
from src.model import PhoBERTSentimentClassifier

# Load trained model
model = PhoBERTSentimentClassifier()
model.load_model('saved_results/final_model')

# Predict sentiment
text = "Thầy giảng rất hay và nhiệt tình"
prediction = model.predict(text)
print(f"Sentiment: {['Negative', 'Neutral', 'Positive'][prediction]}")
```

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.37% |
| **F1 Score** | 94.34% |
| **Precision** | 94.40% |
| **Recall** | 94.37% |

### Dataset Statistics
- **Total Samples**: 16,175
- **Classes**: Negative (46%), Positive (50%), Neutral (4%)
- **Train/Dev/Test**: 70.6% / 9.8% / 19.6%

## 🏗️ System Architecture

### Machine Learning Pipeline
```
Input Text (Vietnamese)
    ↓
PhoBERT Tokenizer (max_length=256)
    ↓
PhoBERT Base Model (135M parameters)
    ↓
RobertaForSequenceClassification
    ↓
3-class Sentiment Output (Negative/Neutral/Positive)
```

### Web Application Stack
```
Frontend (React + Tailwind CSS)
    ↓
RESTful API (FastAPI)
    ↓
ML Model Service (PhoBERT)
    ↓
SQLite Database (Analytics & History)
```

## 📁 Project Structure

```
uit_SentimentAnalysis/
├── 🧠 src/                     # ML Source Code
│   ├── data_preprocessing.py   # Data preparation pipeline
│   ├── model.py               # PhoBERT model architecture
│   ├── train.py               # Training script
│   ├── data_loader.py         # Data loading utilities
│   ├── utils.py               # Helper functions
│   └── visualization.py       # Plotting and analysis
│
├── 🌐 backend/                 # FastAPI Backend
│   ├── app.py                 # Main FastAPI application
│   ├── model_service.py       # ML model integration
│   ├── database.py            # Database operations
│   ├── run.py                 # Server startup script
│   └── requirements.txt       # Python dependencies
│
├── 🎨 frontend/                # React Frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Application pages
│   │   ├── hooks/            # Custom React hooks
│   │   └── utils/            # Frontend utilities
│   ├── public/               # Static assets
│   └── package.json          # Node.js dependencies
│
├── 📊 data/                    # Dataset
│   ├── raw/                   # Original UiT-VSFC data
│   └── processed/             # Preprocessed data
│
├── 📓 notebooks/               # Jupyter Notebooks
│   ├── EDA.ipynb             # Exploratory data analysis
│   └── Evaluation.ipynb      # Model evaluation
│
├── 💾 saved_results/           # Training Outputs
│   ├── final_model/          # Trained PhoBERT model
│   ├── plots/                # Training visualizations
│   └── logs/                 # Training logs
│
└── 🔧 models/                  # Model Configurations
```

## 🔄 Model Training & Replacement

### Training from Scratch

#### 1. Data Preparation
```bash
python src/data_preprocessing.py
```

#### 2. Model Training
```bash
python src/train.py
```

#### 3. Evaluation & Visualization
```bash
python src/visualize_results.py
```

### Training Configuration
```python
{
    "model_name": "vinai/phobert-base",
    "num_labels": 3,
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 20,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch"
}
```

### 🔄 Replacing with Different Models

#### Option 1: Different Pre-trained Models
```python
# In src/model.py, change model_name
MODEL_CONFIGS = {
    "phobert": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large", 
    "bartpho": "vinai/bartpho-word",
    "velectra": "nguyenvulebinh/vi-mrc-base"
}
```

#### Option 2: Custom Model Architecture
```python
# Create new model class in src/model.py
class CustomSentimentClassifier:
    def __init__(self, model_name="your-model"):
        self.model_name = model_name
        # Your custom implementation
    
    def predict(self, text):
        # Your prediction logic
        pass
```

#### Option 3: Integration Steps
1. **Update Model Service**:
   ```python
   # In backend/model_service.py
   from src.model import YourCustomModel
   
   class ModelService:
       def __init__(self):
           self.model = YourCustomModel()
           self.model.load_model('path/to/your/model')
   ```

2. **Update API Response Format**:
   ```python
   # In backend/app.py, modify response structure if needed
   class AnalysisResponse(BaseModel):
       sentiment: str
       confidence: float
       probabilities: Dict[str, float]
       # Add your custom fields here
   ```

3. **Frontend Integration**:
   ```javascript
   // In frontend/src/hooks/useSentimentAnalysis.js
   // Update to handle your model's response format
   ```

## 🌐 Web Application Features

### 📱 Pages & Components

#### 1. **Home Page** (`/`)
- Real-time sentiment analysis
- Input text area with validation
- Results display with confidence scores
- Probability distribution charts

#### 2. **Dashboard** (`/dashboard`)
- Analytics overview
- Sentiment distribution charts
- Usage statistics
- Performance metrics

#### 3. **Analytics** (`/analytics`)
- Detailed analysis history
- Filtering and search capabilities
- Export functionality
- Trend analysis

#### 4. **Admin Panel** (`/admin`)
- System monitoring
- Database management
- Model performance tracking
- User activity logs

### 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | System health check |
| `/analyze` | POST | Sentiment analysis |
| `/analytics` | GET | System analytics |
| `/history` | GET | Analysis history |
| `/admin/stats` | GET | Admin statistics |

### 📊 API Response Examples

#### Sentiment Analysis
```json
{
  "sentiment": "positive",
  "confidence": 0.9234,
  "probabilities": {
    "negative": 0.0123,
    "neutral": 0.0643,
    "positive": 0.9234
  },
  "analysis_id": "uuid-string",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

#### Analytics Data
```json
{
  "total_analyses": 1250,
  "sentiment_distribution": {
    "positive": 45.2,
    "negative": 32.1,
    "neutral": 22.7
  },
  "average_confidence": 0.8756,
  "recent_analyses": [...]
}
```

## 🛠️ Development & Customization

### Backend Development
```bash
# Install development dependencies
cd backend
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
# Install and run development server
cd frontend
npm install
npm start

# Build for production
npm run build
```

### Database Schema
```sql
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    metadata TEXT
);
```

## 🧪 Testing

### Model Testing
```bash
# Run model evaluation
python src/evaluate_model.py

# Test with sample data
python -c "
from src.model import PhoBERTSentimentClassifier
model = PhoBERTSentimentClassifier()
model.load_model('saved_results/final_model')
print(model.predict('Thầy giảng rất hay'))
"
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test analysis endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Thầy giảng rất hay và nhiệt tình"}'
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📈 Performance Optimization

### Model Optimization
- **Model Quantization**: Reduce model size for faster inference
- **ONNX Conversion**: Convert to ONNX for optimized deployment
- **Batch Processing**: Process multiple texts simultaneously

### Backend Optimization
- **Async Processing**: Non-blocking request handling
- **Caching**: Redis integration for frequent requests
- **Load Balancing**: Multiple worker processes

### Frontend Optimization
- **Code Splitting**: Lazy loading of components
- **Caching**: Browser and service worker caching
- **CDN Integration**: Static asset optimization

## 🚀 Deployment

### Local Development
```bash
# Use the provided start script
chmod +x start.sh
./start.sh
```

### Production Deployment
```bash
# Backend
cd backend
pip install -r requirements.txt
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend
cd frontend
npm run build
# Serve build folder with nginx or similar
```

## 📝 Dataset Information

### UiT-VSFC Dataset
- **Source**: University of Information Technology, VNU-HCM
- **Domain**: Vietnamese student feedback on courses and teachers
- **Size**: 16,175 labeled samples
- **Classes**: 
  - **Negative**: 46% (7,399 samples)
  - **Positive**: 50% (8,076 samples) 
  - **Neutral**: 4% (700 samples)
- **Language**: Vietnamese
- **Format**: Text classification

### Data Preprocessing Pipeline
1. **Text Cleaning**: Remove special characters, normalize whitespace
2. **Tokenization**: PhoBERT tokenizer with 256 max length
3. **Label Encoding**: Convert sentiment labels to numerical format
4. **Data Splitting**: Stratified train/validation/test split
5. **Augmentation**: Optional data augmentation techniques

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript code
- Write comprehensive tests
- Update documentation
- Ensure backward compatibility

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PhoBERT Team** (VinAI Research): Excellent Vietnamese language model
- **UiT-VSFC Dataset**: High-quality Vietnamese sentiment dataset
- **Hugging Face**: Transformers library and model hub
- **FastAPI Team**: Modern, fast web framework for APIs
- **React Team**: Powerful frontend library
- **PyTorch Team**: Deep learning framework

## 📧 Contact & Support

**Author**: Huỳnh Lý Tân Khoa  
**Email**: [huynhhlytankhoa@gmail.com](mailto:huynhhlytankhoa@gmail.com)  
**GitHub**: [@coderkhongodo](https://github.com/coderkhongodo)

### Getting Help
- 🐛 **Bug Reports**: [Open an issue](https://github.com/coderkhongodo/uit_sentiment/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/coderkhongodo/uit_sentiment/discussions)
- 📧 **Direct Contact**: Email for urgent matters
- 📚 **Documentation**: Check `/docs` endpoint for API documentation

---

⭐ **Star this repository if you find it helpful!**

🔗 **Links**: [Demo](http://localhost:3000) | [API Docs](http://localhost:8000/docs) | [Dataset](https://github.com/uit-nlp/UiT-VSFC)