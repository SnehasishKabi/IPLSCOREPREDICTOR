# 🏏 IPL Score Predictor with Explainable AI

A comprehensive machine learning application that predicts IPL cricket scores in real-time using **Random Forest** with **SHAP** explanations for transparent AI decision-making.

![Framework](https://img.shields.io/badge/Framework-Streamlit-blue)

This project combines traditional machine learning with explainable AI (XAI) to create a transparent, user-friendly cricket score prediction system. Unlike black-box models, this application explains why it makes each prediction, making it valuable for coaches, analysts, and cricket enthusiasts.

---

## ✨ Key Features
- 🤖 **AI-Powered Predictions**: Random Forest model trained on historical IPL data  
- 🧠 **Explainable AI**: SHAP integration showing exactly what influences each prediction  
- 📊 **Interactive Visualizations**: Feature impact charts, force plots, and contextual insights  
- ⚡ **Real-Time Analysis**: Live match situation processing with instant predictions  
- 🎨 **Professional UI**: Beautiful cricket-themed Streamlit interface  
- 📈 **Performance Metrics**: 85%+ prediction accuracy with comprehensive model evaluation  

---


### Main Interface
- **Team Selection**: Choose batting and bowling teams from 8 consistent IPL franchises  
- **Match Status**: Input current score, wickets, overs, and recent performance  
- **Live Metrics**: Automatic calculation of run rate, strike rate, and remaining resources  

### AI Explanations (SHAP Analysis)
- 📊 **Feature Impact**: Horizontal bar charts showing positive/negative factors  
- 🎯 **Force Plot**: Breakdown of base score vs adjustments  
- 💡 **Key Insights**: Cricket-specific analysis with confidence indicators  

---

## 🛠 Technical Architecture
**Machine Learning Pipeline:**  
`Raw Data → Preprocessing → Feature Engineering → Model Training → SHAP Integration → Deployment`

### Model Performance
| Algorithm        | Train Score | Test Score | Selected |
|------------------|------------|-----------|----------|
| Random Forest    | 97.8%      | 85.2%     | ✅ Best  |
| XGBoost          | 96.1%      | 84.3%     | -        |
| Decision Tree    | 100%       | 76.8%     | -        |
| Linear Regression| 72.4%      | 71.9%     | -        |
| SVM              | 68.2%      | 67.1%     | -        |
| KNN              | 89.3%      | 73.6%     | -        |

  

---

## 📦 Installation
### Prerequisites
- Python 3.8+  
- pip package manager  

### Setup Instructions
```bash
git clone https://github.com/SnehasishKabi/IPLSCOREPREDICTOR.git
cd IPLScorePredictor

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run ipl_score_predictor.py