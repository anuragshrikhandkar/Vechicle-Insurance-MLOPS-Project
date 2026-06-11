# 🚗 Vehicle Insurance Cross-Sell Prediction — End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-FF9900?logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)

> **Predict whether a health-insured customer will also opt for vehicle insurance** — a full production-grade MLOps pipeline from raw data ingestion to a deployed FastAPI web application.

---

## 📌 Problem Statement

An insurance company has data about customers who hold health insurance. The goal is to predict whether these customers would also be interested in purchasing **vehicle insurance**.

By building a predictive model, the company can optimize its outreach strategy, reduce marketing costs, and improve conversion rates.

**Target Variable:** `Response` → `1` (Interested) / `0` (Not Interested)

---

## 🏗️ Project Architecture

```
MongoDB Atlas (Raw Data)
        ↓
Data Ingestion
        ↓
Data Validation  ──→  Validation Report
        ↓
Data Transformation  ──→  Preprocessor (.pkl)
        ↓
Model Training  ──→  Trained Model Artifact
        ↓
Model Evaluation  ──→  Compare vs Production Model
        ↓
Model Pusher  ──→  AWS S3 Bucket
        ↓
FastAPI App (Jinja2 UI)  ──→  /train & /predict endpoints
        ↓
Docker Container  ──→  GitHub Actions CI/CD  ──→  AWS ECR + EC2
```

---

## 📁 Project Structure

```
Vechicle-Insurance-MLOPS-Project/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipline/
│   │   ├── training_pipeline.py       # TrainPipeline class
│   │   └── prediction_pipeline.py     # VehicleData + VehicleDataClassifier
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── configuration/
│   │   └── mongo_db_connection.py
│   │
│   ├── cloud_storage/
│   │   └── aws_storage.py             # boto3 S3 integration
│   │
│   ├── constants/
│   │   └── __init__.py                # APP_HOST, APP_PORT, etc.
│   │
│   └── utils/
│       └── main_utils.py
│
├── config/
│   └── schema.yaml                    # Feature schema for validation
│
├── notebook/
│   └── *.ipynb                        # EDA + experiments
│
├── artifact/                          # Pipeline run artifacts (auto-generated)
│
├── templates/
│   └── vehicledata.html               # Jinja2 prediction form UI
│
├── static/
│   └── css/                           # Frontend styles
│
├── app.py                             # FastAPI application entry point
├── demo.py                            # Pipeline demo/test runner
├── templets.py                        # Project scaffolding script
├── setup.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── workflow.txt
```

---

## 🔢 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `Gender` | int | 0 = Female, 1 = Male |
| `Age` | int | Age of the customer |
| `Driving_License` | int | 0 = No, 1 = Yes |
| `Region_Code` | float | Region code of the customer |
| `Previously_Insured` | int | 0 = No prior vehicle insurance, 1 = Yes |
| `Annual_Premium` | float | Amount paid annually for health insurance |
| `Policy_Sales_Channel` | float | Channel through which policy was communicated |
| `Vintage` | int | Days since customer has been associated |
| `Vehicle_Age_lt_1_Year` | int | Vehicle age < 1 Year (encoded) |
| `Vehicle_Age_gt_2_Years` | int | Vehicle age > 2 Years (encoded) |
| `Vehicle_Damage_Yes` | int | Vehicle was damaged in past (encoded) |
| **`Response`** | int | **Target** — 1 = Interested, 0 = Not Interested |

**Source:** [Kaggle — Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

---

## ⚙️ MLOps Pipeline Components

### 1. 📥 Data Ingestion
- Connects to **MongoDB Atlas** using `pymongo` + `certifi`
- Exports collection to a Pandas DataFrame
- Splits data into train/test sets
- Saves raw CSVs to `artifact/` directory

### 2. ✅ Data Validation
- Validates schema against `config/schema.yaml`
- Checks column names, dtypes, and null thresholds
- Generates a validation report artifact

### 3. 🔄 Data Transformation
- Handles class imbalance using **`imblearn`** (SMOTE / RandomOverSampler)
- Encodes categorical columns (Vehicle_Age, Vehicle_Damage → binary flags)
- Scales numerical features
- Saves preprocessor pipeline as `preprocessor.pkl` using **`dill`**

### 4. 🤖 Model Training
- Trains classifier on transformed data
- Saves trained model as artifact
- Tracks metrics (F1, ROC-AUC, Precision, Recall)

### 5. 📊 Model Evaluation
- Loads existing production model from **AWS S3**
- Compares new model vs production on test set
- Only proceeds if new model beats threshold

### 6. 🚀 Model Pusher
- Pushes accepted model to **AWS S3** bucket using `boto3`
- Makes model available for the prediction pipeline

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Web Framework | FastAPI + Uvicorn |
| Templating | Jinja2 |
| Data Source | MongoDB Atlas (`pymongo`) |
| Cloud Storage | AWS S3 (`boto3`, `mypy-boto3-s3`) |
| ML / Preprocessing | Scikit-learn, imbalanced-learn (`imblearn`) |
| Serialization | `dill` |
| Config | PyYAML, `from_root` |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | AWS EC2 + ECR |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.10+
Docker
AWS CLI configured (with S3 + ECR access)
MongoDB Atlas URI
```

### 1. Clone the Repository
```bash
git clone https://github.com/anuragshrikhandkar/Vechicle-Insurance-MLOPS-Project.git
cd Vechicle-Insurance-MLOPS-Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/<dbname>"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-south-1"
export AWS_S3_BUCKET_NAME="your_bucket_name"
```

### 5. Run Training Pipeline
```bash
python demo.py
```

### 6. Start FastAPI Server
```bash
python app.py
# App runs at http://localhost:8080
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the prediction form (vehicledata.html) |
| `GET` | `/train` | Triggers the full training pipeline |
| `POST` | `/` | Accepts form data, returns prediction result |

### Prediction Response
After form submission, the page renders with one of:
- ✅ `Response-Yes` — Customer is likely interested in vehicle insurance
- ❌ `Response-No` — Customer is not likely interested

---

## 🐳 Docker

```bash
# Build the image
docker build -t vehicle-insurance-mlops .

# Run the container
docker run -p 8080:8080 \
  -e MONGODB_URL="your_mongo_uri" \
  -e AWS_ACCESS_KEY_ID="your_key" \
  -e AWS_SECRET_ACCESS_KEY="your_secret" \
  vehicle-insurance-mlops
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

```
Push to main branch
      ↓
Lint + Unit Tests
      ↓
Docker Build
      ↓
Push image to AWS ECR
      ↓
SSH into EC2 & pull latest image
      ↓
Run container on EC2 (port 8080)
```

**Required GitHub Secrets:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
AWS_ECR_REPO_URI
MONGODB_URL
```

---

## ☁️ AWS Deployment

```
User
  ↓
EC2 Instance (Public IP : 8080)
  ↓
Docker Container (pulled from ECR)
  ↓
FastAPI App
  ↓
Model loaded from S3
  ↓
MongoDB Atlas (data source)
```

### Manual EC2 Deployment
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Pull latest image from ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <ecr-uri>

docker pull <ecr-uri>/vehicle-insurance-mlops:latest

docker run -d -p 8080:8080 \
  -e MONGODB_URL=$MONGODB_URL \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  <ecr-uri>/vehicle-insurance-mlops:latest
```

---

## 📦 Requirements

```
fastapi
uvicorn
jinja2
python-multipart
scikit-learn
pandas
numpy
pymongo
boto3
mypy-boto3-s3
botocore
imblearn
dill
PyYAML
certifi
from_root
matplotlib
seaborn
plotly
ipykernel
-e .
```

---

## 👨‍💻 Author

**Anurag Sandeep Shrikhandkar**
B.Tech — Computer Science (2023–2027)
Prof. Ram Meghe Institute of Technology & Research, Badnera-Amravati

[![GitHub](https://img.shields.io/badge/GitHub-anuragshrikhandkar-black?logo=github)](https://github.com/anuragshrikhandkar)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If this project helped you, drop a star on the repo!
