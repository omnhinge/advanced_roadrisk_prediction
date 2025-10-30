
## 🛣️ Accident Risk Prediction using Machine Learning

### 📘 Overview

This project aims to **predict accident risk levels** based on road, traffic, and environmental conditions.
It uses a **Gradient Boosting Regressor (GBR)** trained on a large dataset of road accidents to output a **risk score** between 0 and 1 (where 1 means high risk).

The project includes:

* A complete **machine learning pipeline** (data preprocessing → model training → evaluation).
* A **Streamlit web app** for real-time risk prediction.
* Handling of **outliers**, **categorical encoding**, and **missing values**.

---

### ⚙️ Features

* 🧹 **Data Preprocessing:**

  * Missing value handling
  * Outlier treatment
  * Label encoding for categorical features
* 🤖 **Model Training:**

  * Gradient Boosting Regressor tuned with optimal hyperparameters
  * Model saved using `joblib`
* 📊 **Model Evaluation:**

  * RMSE, R², and Accuracy Score calculations
* 💻 **Web Interface (Streamlit):**

  * User-friendly interface for manual input
  * Dynamic “Predict Risk” button
  * Outlier-safe prediction
  * Displays risk level: *Low*, *Medium*, or *High*

---

### 🧠 Model Details

| Parameter           | Value | Description                                     |
| ------------------- | ----- | ----------------------------------------------- |
| `learning_rate`     | `0.1` | Step size to prevent overfitting                |
| `n_estimators`      | `100` | Number of boosting stages                       |
| `subsample`         | `0.8` | Introduces randomness for better generalization |
| `max_depth`         | `3`   | Limits tree complexity                          |
| `min_samples_split` | `10`  | Prevents over-splitting                         |
| `min_samples_leaf`  | `4`   | Ensures stable leaf predictions                 |
| `random_state`      | `42`  | Reproducibility                                 |

✅ This configuration provides a **balanced, robust, and generalizable** model.

---

### 📁 Project Structure

```
accident_risk_comp/
│
├── data/
│   └── accidents.csv
│
├── models/
│   ├── final_gradient_boosting_model.pkl
│   └── model_features.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

### 🚀 How to Run

#### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **2️⃣ Train the Model**

If you want to retrain:

```bash
python train_model.py
```

This will:

* Clean and preprocess data
* Train the Gradient Boosting model
* Save both `.pkl` files (model + feature list)

#### **3️⃣ Run the Streamlit App**

```bash
streamlit run app.py
```

#### **4️⃣ Input Values**

Enter road type, weather, lanes, time of day, etc. → click **“Predict Risk”**
You’ll get a **Risk Score** and label (Low / Medium / High).

---

### 🧩 Tech Stack

* **Python 3.11+**
* **Pandas, NumPy** — Data handling
* **Scikit-learn** — Model training & evaluation
* **Joblib** — Model serialization
* **Streamlit** — Web app interface

---

### 📈 Model Performance

| Metric       | Value                              |
| ------------ | ---------------------------------- |
| **R² Score** | ~0.88                              |
| **RMSE**     | Low                                |
| **Accuracy** | ~90% (for categorized risk levels) |

*(Performance varies depending on data split and feature distribution.)*

---

### 🧰 Future Improvements

* Integrate **real-time traffic/weather APIs**
* Implement **ensemble stacking** for even higher accuracy
* Deploy via **Docker + AWS/GCP**
* Add **geospatial accident mapping**

---

### 👨‍💻 Author

**Om **
Machine Learning & Data Engineer
📧 [(omhinge.in@gmail.com)]
💼 [GitHub/omnhinge]

---

