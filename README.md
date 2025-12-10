# MovieInsightML – Revenue Forecasting & Genre Classification

A machine learning system built in Python that predicts **movie revenue** (regression) and classifies **main movie genre** (classification) using a rich combination of text features extracted from movie overviews and continuous numeric attributes. Developed on **Kaggle** as part of a Machine Learning course, this project integrates NLP pipelines with traditional ML models and TinyML deployment.

---

## 🚀 Features

- Mixed-type data pipeline combining text (overview, tagline, keywords) with numeric features (budget, popularity, runtime, vote_average, vote_count)
- Manual TF-IDF vectorization from scratch (no sklearn TfidfVectorizer)
- Feature engineering: word count, average word length, unique word count, sentiment score
- Manual Variance Threshold + Chi-Square feature selection
- Fully manual implementations of core ML algorithms (no sklearn estimators):
  - Decision Tree Classifier & Regressor
  - Random Forest Classifier & Regressor
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Linear Regression
- K-Fold Cross-Validation for robust evaluation
- Regression metrics: MAE, RMSE, R²
- Classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- TensorFlow Lite (TFLite) conversion with benchmarking (size, latency, RAM usage)
- Comparison plots for both classification and regression model performance

---

## 🛠 Technologies Used

- **Language:** Python 3.11
- **Platform:** Kaggle Notebooks
- **Core Libraries:**
  - `numpy`, `pandas` — data manipulation
  - `scipy` — sparse matrix operations, chi-square statistics
  - `matplotlib` — visualization
  - `tensorflow` — TFLite model conversion and inference
- **Algorithms (all manually implemented):**
  - Linear Regression, Decision Tree, Random Forest
  - Logistic Regression, KNN, Decision Tree Classifier, Random Forest Classifier

---

## 📁 Project Structure

```
MovieInsightML/
│
├── Project_Code.ipynb      # Main Jupyter Notebook (full pipeline)
├── .gitignore              # Python/Data Science gitignore
└── README.md               # Project documentation
```

> **Note:** The dataset (`train.csv`, `test.csv`) is sourced from Kaggle and is not included in this repository. See [Dataset](#dataset) section below.

---

## 📊 Dataset

The dataset contains movie records with the following columns:

| Column | Type | Description |
|---|---|---|
| `title` | Text | Movie title |
| `overview` | Text | Movie plot summary |
| `genres` | Text | Pipe-separated genre list |
| `budget` | Numeric | Production budget |
| `revenue` | Numeric | Box office revenue (regression target) |
| `popularity` | Numeric | Popularity score |
| `runtime` | Numeric | Film duration in minutes |
| `vote_average` | Numeric | Average viewer rating |
| `vote_count` | Numeric | Number of votes |
| `tagline` | Text | Movie tagline |
| `keywords` | Text | Associated keywords |
| `credits` | Text | Cast and crew |

- **Train set:** ~274,865 records  
- **Test set:** ~79,533 records

To use this project, download the dataset from [Kaggle](https://www.kaggle.com) and place `train.csv` and `test.csv` in `/kaggle/input/ml-project/` (Kaggle environment) or update the paths in Cell 3 of the notebook accordingly.

---

## ⚙️ How to Run

### On Kaggle (Recommended)
1. Upload `Project_Code.ipynb` to a Kaggle Notebook
2. Attach the dataset as a Kaggle Dataset input (`/kaggle/input/ml-project/`)
3. Run all cells sequentially

### Locally
Ensure you have Python 3.9+ and install dependencies:

```bash
pip install numpy pandas scipy matplotlib tensorflow
```

Update the dataset paths in Cell 3 from:
```python
org_df = pd.read_csv('/kaggle/input/ml-project/train.csv')
org_dt = pd.read_csv('/kaggle/input/ml-project/test.csv')
```
to your local file paths, then run:
```bash
jupyter notebook Project_Code.ipynb
```

---

## 🧠 Pipeline Overview

```
Raw Data
   │
   ├── Text Features (title + overview + tagline + keywords)
   │      ├── Manual TF-IDF (vocabulary: 15,126 terms)
   │      ├── Word Count, Avg Word Length, Unique Word Count
   │      └── Lexicon-Based Sentiment Score
   │
   ├── Numeric Features
   │      ├── budget, popularity, runtime, vote_average, vote_count
   │      └── Min-Max Normalization
   │
   ├── Feature Selection
   │      ├── Variance Threshold (15,126 → 3,702 features)
   │      └── Chi-Square Selection (3,702 → 1,000 features)
   │
   ├── Classification Task (Main Genre)
   │      ├── Manual Logistic Regression
   │      ├── Manual KNN
   │      ├── Manual Decision Tree Classifier
   │      └── Manual Random Forest Classifier
   │
   └── Regression Task (Revenue)
          ├── Manual Linear Regression
          ├── Manual Decision Tree Regressor
          └── Manual Random Forest Regressor
```

---

## 📈 Results Summary

### Classification (Genre Prediction)
| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | — | — |
| KNN | — | — |
| Decision Tree | — | — |
| Random Forest | — | — |

### Regression (Revenue Forecasting)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |

> *Run the notebook to populate actual metric values in your environment.*

---

## 📱 TinyML Deployment

The trained Logistic Regression model is converted to **TensorFlow Lite (.tflite)** format:

- ✅ Quantized model export using `tf.lite.TFLiteConverter`
- ✅ Inference latency benchmarked on simulated edge device
- ✅ Comparison of original vs. TFLite model (size, RAM usage, latency)
- ✅ Compatible with [Edge Impulse](https://edgeimpulse.com) for embedded deployment validation

---

## 🧪 Concepts Practiced

This project was developed during the **Machine Learning** course and applies the following advanced concepts:

- Mixed-type data handling (text + continuous numeric)
- Text preprocessing and TF-IDF vectorization from scratch
- Feature engineering from unstructured text
- Manual Chi-Square and Variance Threshold feature selection
- Fully manual ML algorithm implementations (no sklearn estimators)
- K-Fold Cross-Validation
- Regression and classification tasks in a single pipeline
- Model interpretation via metrics and comparison plots
- TensorFlow Lite conversion and TinyML deployment feasibility

---

## 👨‍💻 Author

**Ehtisham Abid**
