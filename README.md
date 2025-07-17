# 🌍 QuakeSight: An End-to-End Analytics Framework for Earthquake Risk Mitigation
An end-to-end analytics framework for real-time earthquake prediction, risk assessment, and disaster mitigation using machine learning and geospatial intelligence.
[![Python](https://img.shields.io/badge/Built%20With-Python-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Powered%20By-XGBoost-green?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)]



**QuakeSight** is a comprehensive earthquake disaster management system that combines real-time data collection, machine learning-based prediction, and prescriptive analytics to forecast earthquake risks and guide proactive response planning. The framework integrates all four types of analytics — **Descriptive, Diagnostic, Predictive, and Prescriptive** — to enable informed decision-making using geospatial and temporal earthquake data.

---

## 📌 Key Features

- ⏱️ **Real-time data sourcing** from the [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/).
- 🧠 **Predictive modeling** using **XGBoost** to forecast earthquake epicenters for the next 7 days.
- 🗺️ **Interactive visualization** of historical and predicted earthquakes on a live map.
- 📊 **Descriptive & Diagnostic analysis** done in Jupyter Notebooks for EDA and feature engineering.
- ⚙️ **Prescriptive strategies** based on spatial clustering and historical frequency zones.
- 🚀 **Deployed as a Streamlit web app** for easy access and real-time predictions.

---

## 🔍 Techniques & Implementation

### 🛰️ Real-time Data Sourcing
- We fetch **live seismic activity data** using the USGS Earthquake Catalog API in GeoJSON format.
- The system pulls attributes like **latitude, longitude, depth, magnitude, time**, and **location type**.

### 🔁 Rolling Window Forecasting
- Earthquake data is **time-series and irregular**, so we apply **rolling windows** to frame prediction as a supervised learning task.
- For each time window (e.g., past 7 days), we predict the most probable **epicenter coordinates** and **magnitude** for the **next 7 days**.
- This captures both **short-term trends** and **temporal-spatial patterns** in seismic activity.

### 🧠 Predictive Modeling
- Model used: **XGBoost Regressor** for predicting future **latitude**, **longitude**, and **magnitude**.
- Features include rolling statistical measures (mean, std), frequency zones, and recent activity hot zones.
- Train-Test split is performed in chronological order to prevent data leakage.
- Output: A prediction map showing **probable epicenters** and **confidence levels**.

### 📊 Descriptive & Diagnostic Analytics
- Performed in **Jupyter Notebooks** using Python libraries like `pandas`, `matplotlib`, `seaborn`, and `geopandas`.
- Includes:
  - Trend analysis of magnitude vs depth.
  - Geographic clustering using **DBSCAN** and **K-Means**.
  - Correlation analysis of time-lag and location-based features.

### 🚀 Predictive & Prescriptive Interface
- Implemented using **Streamlit**, where:
  - Users can see **real-time predictions** on an interactive map (Folium).
  - Get recommendations like **high-risk zones**, **magnitude probabilities**, and **time windows to monitor**.
  - Prescriptive actions are visualized using **heatmaps** and **alert zones**.

---

## 🧰 Tech Stack

| Layer              | Technologies Used                                           |
|-------------------|-------------------------------------------------------------|
| Data Source        | USGS Real-time Earthquake API                               |
| Data Processing    | Python, Pandas, NumPy                                       |
| EDA & Clustering   | Jupyter Notebook, Seaborn, Matplotlib, GeoPandas, DBSCAN    |
| Machine Learning   | XGBoost, Scikit-learn                                       |
| Web Interface      | Streamlit, Folium, Plotly                                   |
| Deployment         | Streamlit Cloud / Local Server                              |

---
## 🤝 Contributing

Feel free to open issues, submit PRs, or suggest improvements. This project aims to empower disaster preparedness through open-source innovation.

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🌐 References

- [USGS Earthquake API Documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Folium – Python Data, Leaflet.js Maps](https://python-visualization.github.io/folium/)

