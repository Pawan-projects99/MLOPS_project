## End to end Machine learning  project
📊 Project Overview
This is a production-ready MLOps implementation for predicting student math performance based on various demographic and academic factors. The project demonstrates modern machine learning operations practices including experiment tracking, model versioning, containerization, and automated deployment.
🎯 Problem Statement
Predict student math scores based on:

Gender and race/ethnicity
Parental education level
Lunch program participation
Test preparation course completion
Reading and writing scores

🏗️ MLOps Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │ -> │  DVC Pipeline   │ -> │  MLflow Track   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  FastAPI Serve  │ <- │  Best Model     │ <- │ Model Registry  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         v
┌─────────────────┐
│ Docker + AWS    │
│   Deployment    │
└─────────────────┘
🚀 Quick Start
Prerequisites

Python 3.8+
Docker (optional)
AWS CLI (for cloud deployment)

1. Clone and Setup
bashgit clone https://github.com/your-username/student-performance-mlops.git
cd student-performance-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
2. Quick Demo
bash# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Train models (in another terminal)
make train

# Start API server
make serve

# Visit http://localhost:8000/docs for API documentation
3. Make a Prediction
bashcurl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "male",
       "race_ethnicity": "group A",
       "parental_level_of_education": "bachelor degree",
       "lunch": "standard",
       "test_preparation_course": "completed",
       "reading_score": 85,
       "writing_score": 80
     }'
🛠️ MLOps Features
1. Data Version Control (DVC)

Data Versioning: Track data changes and model artifacts
Pipeline Automation: Reproducible ML workflows
Remote Storage: S3 integration for artifact storage

bash# Run complete pipeline
dvc repro

# Push artifacts to remote storage
dvc push
2. Experiment Tracking (MLflow)

Experiment Management: Track parameters, metrics, and models
Model Registry: Version control for production models
Model Comparison: Compare different model versions

Access MLflow UI: http://localhost:5000
3. Model Serving (FastAPI)

REST API: Production-ready model serving
Interactive Documentation: Auto-generated API docs
Input Validation: Pydantic schemas for type safety
Health Checks: Built-in monitoring endpoints

API Documentation: http://localhost:8000/docs
4. Containerization (Docker)
bash# Build serving container
docker build -f docker/Dockerfile.serve -t student-performance:latest .

# Run container
docker run -p 8000:8000 student-performance:latest

# Full stack with docker-compose
cd docker && docker-compose up
5. Infrastructure as Code (Terraform)
Deploy to AWS with ECS, ECR, and Application Load Balancer:
bashcd terraform
terraform init
terraform plan
terraform apply
6. CI/CD Pipeline (GitHub Actions)
Automated workflows for:

## ✅ Code quality checks (linting, testing)
🏗️ Model training and validation
🐳 Docker image building
🚀 AWS deployment
📊 Performance monitoring

## 📊 Model Performance
ModelR² ScoreMAERMSERandom Forest0.8655.236.87Gradient Boosting0.8515.457.12Linear Regression0.7846.788.45
Best Model: Random Forest

## 📈 API Endpoints
Core Endpoints

GET / - API information
GET /health - Health check
POST /predict - Single prediction
POST /batch_predict - Batch predictions
GET /model/info - Model information

## Example Request/Response
Request:
json{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "master's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 90,
  "writing_score": 88
}
Response:
json{
  "predicted_math_score": 87.5,
  "model_confidence": "High",
  "prediction_timestamp": "2025-09-08T10:30:00",
  "input_features": {...}
}
## 🏗️ AWS Deployment Architecture
Internet -> ALB -> ECS Fargate -> ECR
              |
              v
         CloudWatch (Monitoring)
              |
              v
         SNS (Alerts) -> Email
Deployed Components

ECS Cluster: Container orchestration
Application Load Balancer: Traffic distribution
ECR Repository: Container registry
CloudWatch: Logging and monitoring
SNS: Alert notifications
S3: Artifact storage

## 📊 Monitoring & Alerts
Built-in Monitoring

Data Drift Detection: Monitor input data distribution changes
Model Performance: Track prediction accuracy over time
System Health: API response times and error rates
Resource Usage: CPU, memory, and network metrics

Alert Conditions

Model performance degradation (R² < 0.8)
High API error rate (> 5%)
Resource utilization (CPU > 80%, Memory > 80%)
Data drift detection triggers

## 🔄 Project Status: Production Ready ✅
This project demonstrates enterprise-grade MLOps practices and is suitable for production deployment with proper configuration and monitoring.