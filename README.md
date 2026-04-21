# 🚔 Crime Prediction Dashboard

### Fairness-Constrained Multi-Task Learning (FC-MT-LSTM)

A **final-year research project** focused on predicting crime patterns across vulnerable groups in India using deep learning, while ensuring fairness across groups.

🔗 **Live Demo:** https://nehulagarwal.github.io/crime-dashboard-

---

## 📌 Project Overview

This project builds a **fairness-aware crime prediction system** using a custom deep learning architecture (**FC-MT-LSTM**) and visualizes results through an interactive React dashboard.

The model predicts crime rates for:

* **SC (Scheduled Castes)**
* **ST (Scheduled Tribes)**
* **Women**
* **Children**

Unlike traditional models, this system ensures **balanced performance across all groups**, reducing bias in predictions.

---

## 🎯 Key Features

* 🔥 Custom Deep Learning Model (FC-MT-LSTM V5 Enhanced)
* ⚖️ Fairness-aware loss function
* 📊 Interactive dashboard with multiple visualizations
* 📈 Comparison with 6 baseline models
* 🌍 State-wise and group-wise analysis
* 🔮 Future-ready prediction pipeline

---

## 🧠 Model Highlights

* Multi-task learning (4 parallel outputs)
* Residual architecture + LayerNorm
* Fairness constraint in loss function:

[
L = L_{MSE} + \lambda \cdot L_{fairness}
]

* Achieves:

  * **R²:** 0.9980
  * **MAE:** 3.79
  * **Fairness Ratio:** 3.26

---

## 📊 Dataset

* Source: NCRB (National Crime Records Bureau, India)
* Years: **2017 – 2022**
* Records: **21,067**
* Features: **188**

### Data Split

* **Training:** 2017–2021
* **Testing:** 2022

---

## 🖥️ Dashboard Pages

* **Home:** Project overview & KPIs
* **Overview:** Dataset statistics
* **Trends:** Year-wise crime trends
* **Cities:** Metro city analysis
* **Predictions:** Model outputs (scatter, state comparison, records)
* **Models:** Comparison with baseline models
* **Fairness:** Bias analysis across groups

---

## ⚙️ Tech Stack

### Backend (ML)

* Python
* PyTorch
* Pandas, NumPy, Scikit-learn

### Frontend

* React 19
* Recharts
* React Router

### Deployment

* GitHub Pages

---

## 📁 Project Structure

```
crime-dashboard/
├── src/
│   ├── pages/
│   ├── components/
│   ├── data/
│   └── App.js
├── public/
├── package.json
```

---

## 🚀 How to Run

### 1. Clone repo

```
git clone https://github.com/nehulagarwal/crime-dashboard-.git
cd crime-dashboard
```

---

### 2. Install dependencies

```
npm install --legacy-peer-deps
```

---

### 3. Run frontend

```
npm start
```

---

### 4. Run backend (optional)

```
cd src/data
python run_predictions.py
```

---

## 📈 Results (Paper)

| Metric         | Value  |
| -------------- | ------ |
| MAE            | 3.79   |
| RMSE           | 9.83   |
| R²             | 0.9980 |
| Fairness Ratio | 3.26   |

---

## ⚠️ Notes

* Paper results are shown in dashboard for consistency
* Live model outputs are also generated via Python pipeline
* Use Python **3.10/3.11** for PyTorch compatibility

---

## 👨‍💻 Team

* Nehul Agarwal
* Rishabh Singh
* Farah Masoodi
* Archit Raghav

**Guide:** Prof. Geetanjali Tyagi


Give it a ⭐ on GitHub!
