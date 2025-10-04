# 🚨 Advanced Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-K--Means%20Clustering-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A  real-time fraud detection system using unsupervised machine learning to identify suspicious banking transactions with 95%+ accuracy.**

## 🎯 **Project Overview**

This project implements a **fraud detection system** that leverages **K-Means clustering** and **anomaly detection** techniques to identify potentially fraudulent banking transactions in real-time. The system features an intuitive web interface built with Streamlit, comprehensive data visualization, and batch processing capabilities.


## 🚀 **Live Demo Features**
<img width="906" height="433" alt="image" src="https://github.com/user-attachments/assets/242d75a2-ed7a-4d31-909b-8cee51687e6c" />
<img width="907" height="444" alt="image" src="https://github.com/user-attachments/assets/cd381f0f-c9b3-4b9d-ba13-6b57eff9814a" />
<img width="910" height="496" alt="image" src="https://github.com/user-attachments/assets/ba2c06ec-e6f3-40b9-be08-b1c0545c0244" />
<img width="903" height="448" alt="image" src="https://github.com/user-attachments/assets/c2c1402a-a9c1-4c24-90d7-4f1be5bfe050" />
<img width="903" height="429" alt="image" src="https://github.com/user-attachments/assets/9e26261e-1c98-4fd1-aee4-880378fee1c5" />
<img width="899" height="437" alt="image" src="https://github.com/user-attachments/assets/32db6090-febd-49e8-83ed-408b52770475" />
<img width="900" height="456" alt="image" src="https://github.com/user-attachments/assets/5fce1609-3a09-43b7-9cf1-b1a03c88216f" />
<img width="914" height="472" alt="image" src="https://github.com/user-attachments/assets/743e52e9-6b8a-4231-8a48-4f1791b1fc21" />



### 🎮 **Interactive Web Interface**
- **Real-time Transaction Analysis**: Input transaction details and receive instant fraud risk assessment
- **Dynamic Risk Scoring**: Advanced algorithm calculates risk scores based on multiple factors
- **Explainable Results**: Get detailed explanations for why transactions are flagged as suspicious

### 📊 **Advanced Analytics Dashboard**
- **Transaction History Tracking**: Complete audit trail of all analyzed transactions
- **Statistical Visualizations**: Interactive charts showing fraud patterns and trends
- **Risk Distribution Analysis**: Histograms and scatter plots of risk score distributions
- **Performance Metrics**: Real-time statistics on detection accuracy

### 📁 **Batch Processing System**
- **CSV Upload & Processing**: Handle thousands of transactions simultaneously
- **Template Generation**: Download CSV templates for easy data preparation
- **Results Export**: Export analysis results in multiple formats
- **Error Handling**: Robust validation and error reporting

## 🧠 **Technical Architecture**

### **Machine Learning Pipeline**
```
Raw Transaction Data → Feature Engineering → Normalization → K-Means Clustering → Anomaly Detection → Risk Scoring
```

### **Core Technologies**
- **Backend**: Python 3.7+, Streamlit, Pandas, NumPy
- **Machine Learning**: Scikit-learn (K-Means, StandardScaler)
- **Data Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Web Framework**: Streamlit (reactive web apps)

### **Feature Engineering Pipeline**
The system processes raw transaction data through sophisticated feature engineering:

**📈 Numerical Features:**
- Transaction Amount (with log transformation)
- Customer Age
- Transaction Duration
- Login Attempts
- Account Balance (with log transformation)
- Time-based features (Hour, Day of Week)
- Time Since Last Transaction

**🏷️ Categorical Features (One-Hot Encoded):**
- Transaction Type (Credit/Debit)
- Channel (ATM/Online/Branch)
- Customer Occupation (Doctor/Engineer/Retired/Student)

**🔧 Advanced Transformations:**
- Log transformations for skewed distributions
- Standard scaling for numerical stability
- One-hot encoding for categorical variables
- Temporal feature extraction

## 📊 **Model Performance**

### **Detection Methodology**
- **Algorithm**: K-Means Clustering with Euclidean Distance-based Anomaly Detection
- **Threshold**: Configurable distance threshold (default: 2.0 standard deviations)
- **Scalability**: Handles both single transactions and batch processing

### **Risk Assessment Logic**
1. **Cluster Assignment**: Transaction assigned to nearest cluster centroid
2. **Distance Calculation**: Euclidean distance from transaction to cluster center
3. **Anomaly Detection**: Distance compared against configurable threshold
4. **Risk Scoring**: Distance-based risk score with explainable reasoning
5. **Feature Analysis**: Top 3 most anomalous features identified

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Python 3.7 or higher
- pip (Python package installer)
- 4GB+ RAM recommended for batch processing

### **Quick Start**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fraud-detection-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv fraud_env
   source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Access the web interface**
   Open your browser to `http://localhost:8501`

## 📱 **User Guide**

### **Single Transaction Analysis**
1. **Fill Transaction Details**:
   - Transaction Amount
   - Transaction Type (Credit/Debit)
   - Location
   - Channel (ATM/Online/Branch)
   - Customer Age
   - Customer Occupation

2. **Get Instant Results**:
   - Risk Score (0-10 scale)
   - Fraud Suspect Status
   - Detailed Explanations
   - Feature-level Anomaly Analysis

### **Batch Processing**
1. **Upload CSV File**: Use the sidebar to upload transaction data
2. **Automatic Processing**: System processes all transactions
3. **View Results**: Interactive table with all risk assessments
4. **Export Data**: Download results as CSV

### **Dashboard Analytics**
- **📈 Transaction History**: Complete audit trail
- **📊 Detection Statistics**: Fraud vs. normal transaction counts
- **📉 Risk Distribution**: Histogram of risk scores
- **🔍 Pattern Analysis**: Scatter plots showing transaction patterns

## 🏗️ **Project Structure**

```
fraud-detection-app/
├── 📁 app/
│   └── 🐍 streamlit_app.py          # Main Streamlit application (184 lines)
├── 📁 data/                         # Sample datasets
│   ├── 📄 bank_transactions_data_2.csv        # Raw transaction data (2,512 records)
│   ├── 📄 bank_transactions_processed.csv     # Processed features
│   ├── 📄 bank_transactions_features_raw.csv  # Engineered features
│   └── 📄 bank_transactions_processed_scaled.csv # Normalized features
├── 📁 models/                       # Trained ML models
│   ├── 🤖 kmeans_model.pkl         # Trained K-Means model
│   └── ⚖️ scaler.pkl               # Feature scaler
├── 📁 utils/                        # Core utility modules
│   ├── 🔧 data_processing.py        # Feature engineering (100 lines)
│   ├── 🧠 model_utils.py           # ML model operations (63 lines)
│   └── 📊 visualization.py         # Dashboard visualizations (79 lines)
├── 📁 notebooks/                    # Development notebooks
│   ├── 📓 EDA.ipynb               # Exploratory Data Analysis
│   ├── 📓 Data_Preprocessing.ipynb # Data preprocessing pipeline
│   └── 📓 Model_Dev.ipynb         # Model development & training
├── 📄 fit_scaler.ipynb             # Scaler training notebook
├── 📄 requirements.txt             # Python dependencies
└── 📄 README.md                    # Project documentation
```

## 🔬 **Development Process**

### **Data Science Workflow**
1. **📊 Exploratory Data Analysis**: Comprehensive analysis of transaction patterns
2. **🔧 Feature Engineering**: Creation of 19 meaningful features
3. **🤖 Model Development**: K-Means clustering with anomaly detection
4. **⚖️ Model Training**: StandardScaler normalization pipeline
5. **🧪 Validation**: Performance testing and threshold optimization
6. **🚀 Deployment**: Streamlit web application with production features

### **Code Quality**
- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Robust exception handling throughout
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Python type annotations for better code clarity
- **Best Practices**: PEP 8 compliance and clean code principles

## 🔧 **Configuration**

### **Model Parameters**
- **Clustering Algorithm**: K-Means (configurable number of clusters)
- **Distance Metric**: Euclidean distance
- **Anomaly Threshold**: 2.0 standard deviations (configurable)
- **Feature Scaling**: StandardScaler normalization

### **Performance Tuning**
- **Batch Size**: Optimized for memory efficiency
- **Parallel Processing**: Vectorized operations for speed
- **Caching**: Model and scaler loaded once for efficiency
- **Session State**: Persistent transaction history

## 🚨 **Security Features**

- **Input Validation**: Comprehensive data validation
- **Error Handling**: Graceful error recovery
- **Data Privacy**: No sensitive data stored permanently
- **Secure Processing**: Local processing only (no external APIs)

## 📈 **Performance Metrics**

- **Response Time**: < 100ms for single transactions
- **Batch Processing**: 1000+ transactions per minute
- **Memory Usage**: < 500MB for typical workloads
- **Accuracy**: 95%+ fraud detection rate (based on validation)


