# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using classification algorithms. Built as part of a data science portfolio to demonstrate end-to-end ML workflow — from raw data to model comparison and feature importance analysis.

---

## 📌 Project Overview

The Titanic dataset is one of the most well-known datasets in data science. This project explores the question:

> **"What factors determined whether a passenger survived the Titanic disaster?"**

Using real passenger data, three classification models are trained, evaluated, and compared to identify the best predictor of survival.

---

## 📁 Dataset

- **File:** `Titanic-Dataset.csv`
- **Source:** [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Records:** 891 passengers
- **Target column:** `Survived` (0 = Did not survive, 1 = Survived)

---

## 🔧 Project Workflow

### 1. Exploratory Data Analysis
- Loaded and inspected the dataset (shape, dtypes, descriptive stats)
- Identified missing values across columns

### 2. Data Preprocessing
- **Dropped** the `Cabin` column due to excessive missing values (~77%)
- **Filled** missing `Age` values with the **median** (robust to outliers)
- **Filled** missing `Embarked` values with the **mode** (most frequent category)
- **Dropped** non-predictive columns: `Name`, `Ticket`, `PassengerId`

### 3. Encoding
- Applied **One-Hot Encoding** via `pd.get_dummies()` on `Sex` and `Embarked`
- Used `drop_first=True` to avoid the dummy variable trap

### 4. Visualisation
- Age distribution histogram — majority of passengers were aged 20–30
- Countplot of gender vs survival — females had significantly higher survival rates

### 5. Train-Test Split
- **80% training / 20% testing** split using `train_test_split` with `random_state=42`

### 6. Model Training & Evaluation
Three models were trained and compared:

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~80% | Strong baseline; better at predicting deaths than survivors |
| Decision Tree (depth=3) | ~78% | Simplified tree; lower recall due to max_depth constraint |
| **Random Forest (100 trees)** | **~82%** | **Best overall accuracy and recall** |

### 7. Feature Importance
Using the Random Forest model, the most influential features were identified:

> *"Survival was primarily influenced by **gender**, **fare (wealth)**, and **age**. Females and higher-paying passengers had significantly higher survival rates, aligning with historical 'women and children first' evacuation policies."*

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading & preprocessing |
| Matplotlib / Seaborn | Data visualisation |
| Scikit-learn | Model training & evaluation |

---

## 📊 Key Findings

- **Gender** was the strongest predictor of survival — females survived at a much higher rate
- **Fare** acted as a proxy for passenger class and wealth, with higher-paying passengers more likely to survive
- **Age** also played a role, with younger passengers having better odds
- Random Forest outperformed both Logistic Regression and Decision Tree on this dataset

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/akashanil4/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn jupyter
   ```

3. Place `Titanic-Dataset.csv` in the project root directory.

4. Launch the notebook:
   ```bash
   jupyter notebook Titanic_Survival_prediction_.ipynb
   ```

---

## 📂 Project Structure

```
titanic-survival-prediction/
│
├── Titanic_Survival_prediction_.ipynb   # Main notebook
├── Titanic-Dataset.csv                  # Dataset (download from Kaggle)
└── README.md                            # Project documentation
```

---

## 👤 Author

**Akash Anil**
- GitHub: [@akashanil4](https://github.com/akashanil4)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
