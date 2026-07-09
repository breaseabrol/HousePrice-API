# House Price Prediction API
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready machine learning API for predicting housing prices using XGBoost, deployed on Kubernetes with FastAPI.

## 🎯 Overview

This project demonstrates end-to-end ML engineering: from data preprocessing and feature engineering to model training, API development, and cloud deployment. The system achieves **0.62 R² score** on test data with automated real-time predictions.

## ✨ Features

- **XGBoost ML Model** – Trained on housing dataset with advanced feature engineering
- **FastAPI REST Endpoint** – Real-time predictions with Pydantic validation
- **Kubernetes Deployment** – Containerized microservices with load balancing and fault tolerance
- **Automated Feature Engineering** – Interaction terms, amenity aggregation, luxury index calculation
- **Model Persistence** – Joblib serialization for reproducible inference across deployments

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² Score (Test) | 0.62 |
| Mean Squared Error | 1.20e12 |
| Features Used | 16 (12 input + 4 engineered) |

## 📊 Dataset

This project uses the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) 
from Kaggle, released under CC0 (public domain).

## 🗃️ Model Artifacts

`model.pkl` and `scaler.pkl` are intentionally excluded from version control
(see `.gitignore`) — binary model artifacts don't belong in git history, as
they bloat repo size and cause unnecessary merge conflicts on every retrain.

To regenerate them locally, run the training workflow in `Project.ipynb`.
The notebook will output `model.pkl` and `scaler.pkl` into the project root,
which `main.py` loads on startup.

> **Note:** You must generate these files before running the API locally —
> see [Quick Start](#-quick-start) below.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│   Client Request (JSON)                 │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│   FastAPI Endpoint (/predict)           │
│   - Pydantic Validation                 │
│   - Feature Engineering                 │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│   XGBoost Model + MinMaxScaler          │
│   - Real-time Inference                 │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│   JSON Response (Predicted Price)       │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker
- Kubernetes (optional, for deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/breaseabrol/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Generate Model Artifacts

`model.pkl` and `scaler.pkl` are not tracked in this repo (see
[Model Artifacts](#️-model-artifacts)) and must be generated before the API
will run:

```bash
# Run all cells in Project.ipynb to train the model and save
# model.pkl and scaler.pkl into the project root
jupyter notebook Project.ipynb
```

### Run Locally

```bash
# Start FastAPI server
uvicorn main:app --reload --port 8080

# API will be available at http://localhost:8080
# Interactive docs: http://localhost:8080/docs
```

### Example Request

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 7420,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 3,
    "parking": 2,
    "furnishingstatus": 2,
    "mainroad_yes": 1,
    "guestroom_yes": 0,
    "basement_yes": 0,
    "hotwaterheating_yes": 0,
    "airconditioning_yes": 1,
    "prefarea_yes": 1
  }'
```

**Response:**
```json
{
  "prediction": 13250000.5
}
```

## 📦 Deployment

### Docker

```bash
# Build image
docker build -t house-price-api:latest .

# Run container
docker run -p 8080:8080 house-price-api:latest
```

### Kubernetes

```bash
# Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment
kubectl get pods
kubectl get svc

# Access via LoadBalancer
kubectl port-forward svc/data-api-service 8080:80
```

## 📂 Project Structure

```
house-price-prediction/
├── main.py                 # FastAPI application
├── model.pkl                # Trained XGBoost model (generated, gitignored)
├── scaler.pkl                # MinMaxScaler for normalization (generated, gitignored)
├── requirements.txt        # Python dependencies
├── Dockerfile               # Container configuration
├── deployment.yaml          # Kubernetes deployment
├── service.yaml              # Kubernetes service
├── Project.ipynb             # Jupyter notebook (model training)
├── .gitignore                 # Excludes model artifacts, caches, env files
└── README.md                # This file
```

## 🔧 Model Details

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `area_bedrooms` | Interaction term: area × bedrooms |
| `stories_bathrooms` | Interaction term: stories × bathrooms |
| `amenities_count` | Sum of 6 binary amenity features |
| `luxury_index` | Sum of AC, heating, guest room, basement |

### Data Preprocessing

- **MinMaxScaler** – Normalizes area to [0, 1] range
- **One-Hot Encoding** – Converts binary features (mainroad, guestroom, etc.)
- **Outlier Removal** – IQR-based filtering on price

### Model Hyperparameters

```python
XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

## 📈 Training Pipeline

See `Project.ipynb` for the complete training workflow:

1. Data Loading & Exploration
2. Feature Engineering
3. Train-Test Split (70-30)
4. Model Training & Validation
5. Model & Scaler Serialization (outputs `model.pkl` and `scaler.pkl`)

## 🛠️ Technologies Used

- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
- **API Framework:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **Serialization:** Joblib

## 📝 API Documentation

### Endpoint: `/predict`

**Method:** POST

**Request Body:**
```json
{
  "area": float,
  "bedrooms": int,
  "bathrooms": int,
  "stories": int,
  "parking": int,
  "furnishingstatus": int,
  "mainroad_yes": int (0 or 1),
  "guestroom_yes": int (0 or 1),
  "basement_yes": int (0 or 1),
  "hotwaterheating_yes": int (0 or 1),
  "airconditioning_yes": int (0 or 1),
  "prefarea_yes": int (0 or 1)
}
```

**Response:**
```json
{
  "prediction": float
}
```

**Error Response:**
```json
{
  "error": "Error message"
}
```

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Feature engineering and data preprocessing
- ✅ Model training and evaluation
- ✅ REST API design with FastAPI
- ✅ Containerization with Docker
- ✅ Kubernetes microservices deployment
- ✅ Production-grade model persistence, with binary artifacts kept out of version control

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via Issues
- Submit pull requests for improvements
- Suggest new features

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Branden Rease Abrol**
- GitHub: [@breaseabrol](https://github.com/breaseabrol)
- LinkedIn: [breaseabrol](https://linkedin.com/in/breaseabrol)
- Email: brandenabrol6805@gmail.com


---

**⭐ If this project helped you, please consider giving it a star!**
