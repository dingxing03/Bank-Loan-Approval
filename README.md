# 📊 Bank Loan Approval Prediction

A machine learning project for predicting **bank loan approvals** using multiple classification models. The project explores data preparation, model training, evaluation, and deployment to assist financial institutions in making **data-driven lending decisions**.

---

## 📌 Project Overview
This project uses the **BankLoanApproval.csv** dataset (255,327 records) and evaluates whether new loan applications should be approved or rejected. Various **classification algorithms** were implemented and compared to find the most effective model.  

**Key Objective:**  
> Predict loan approval outcomes (approve/reject) for new applicants using supervised machine learning models.

---

## ⚙️ Workflow
### 1. Data Preparation
- **Exploration:** Checked data types, descriptive statistics, and visualizations (loan amount, employment type, etc.).  
- **Splitting:** Train (70%), Test (20%), Validation (10%).  
- **Preprocessing:**  
  - Removed unused `LoanID` column  
  - One-hot encoded categorical variables  
  - Handled missing values, duplicates, and outliers  
  - Normalized features using `StandardScaler`  

### 2. Models Implemented
- k-Nearest Neighbors (kNN)  
- Logistic Regression  
- Support Vector Machine (SVM) with kernels: linear, RBF, polynomial, sigmoid  
- Decision Trees (CART)  
- Random Forest  
- Naive Bayes (Gaussian)  
- Neural Network  

### 3. Model Evaluation
- **Metrics Used:** Accuracy, Precision, Recall, F1-score, ROC Curve & AUC Score  
- **Findings:**  
  - Models performed well for approved loans (majority class).  
  - Struggled with rejected loans due to class imbalance.  
  - **Neural Network achieved the best overall performance** with the highest AUC (0.75) and improved F1-score for rejected loans.  

### 4. Prediction on New Applicants
- Applied the best model (**Neural Network**) to **20 new applicants**.  
- Result: **18 approved, 2 rejected**.  

---

## 📊 Results Summary
| Model              | F1-score (Class 0) | F1-score (Class 1) | AUC Score |
|---------------------|-------------------|-------------------|------------|
| kNN                | 0.94              | 0.01              | 0.68       |
| Logistic Regression | 0.94              | 0.07              | 0.75       |
| SVM (Linear)       | 0.94              | 0.00              | 0.49       |
| SVM (RBF)          | 0.94              | 0.00              | 0.64       |
| Decision Trees     | 0.94              | 0.04              | 0.71       |
| Random Forest      | 0.94              | 0.06              | 0.74       |
| Naive Bayes        | 0.94              | 0.06              | 0.75       |
| **Neural Network** | **0.94**          | **0.12**          | **0.75**   |

---

## 🚀 Tech Stack
- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Machine Learning**: classification models, hyperparameter tuning with GridSearchCV / RandomizedSearchCV  
- **Visualization**: EDA plots, ROC & AUC curves, confusion matrices  

---

## 🛠️ Installation & Usage
### Prerequisites
- Python 3.8+  
- Jupyter Notebook (or Google Colab)  

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dingxing03/Bank-Loan-Approval.git
   cd bank-loan-approval
