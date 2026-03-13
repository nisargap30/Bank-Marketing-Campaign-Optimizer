# Bank Marketing Campaign Optimizer

Predicting which customers are likely to subscribe to a term deposit, using telemarketing data from a Portuguese bank (2008–2013). Built as part of a Statistical Methods course project.

The short version: The bank was calling everyone. Most calls failed. This project builds a model to figure out *who's actually worth calling* before picking up the phone.

---

## The Problem

The dataset covers 41,188 customer contacts with an 11.27% success rate — meaning roughly 9 out of 10 calls converted nobody. The goal is to build a binary classifier that predicts subscription likelihood *before* a call is made, so campaign managers can prioritize high-probability customers.

One non-obvious constraint: **call duration is excluded from all models**. It's highly correlated with outcome, but it's only known after the call ends — using it would be data leakage.

---

## What's in This Repo

```
bank-marketing-optimizer/
│
├── notebook/
│   └── Stats_Mod_Proj.ipynb       # Main analysis notebook (EDA + modeling)
│
├── data/
│   └── README.md                  # Instructions for downloading the dataset
│
├── Plots/
│   └── plots.png
│
├── requirements.txt
└── README.md
```

---

## Dataset

The data is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) — specifically the full version (`bank-additional-full.csv`, semicolon-separated).

**Download it manually:**
1. Go to https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Click **Download**
3. Unzip and grab `bank-additional-full.csv`
4. If running locally, place it in the `data/` folder
5. If running on Colab, upload it when prompted (the notebook handles this)

The file is not included in this repo due to UCI's terms of use.

---

## Key Findings

- **Economic conditions dominate everything else.** The Euribor 3-month rate is the strongest single predictor — when rates are low (post-2008 crisis), customers flee to safe savings instruments. Demographics barely move the needle by comparison.
- **Cellular contacts convert at 14.7% vs. 5.2% for telephone.** Most of the bank's high-volume months are also its worst-performing ones.
- **Students (31.4%) and retirees (25.2%) convert at far higher rates** than the contact-heavy 30–50 age band.
- **Credit default = automatic disqualifier.** Zero conversions observed across the entire dataset for customers with default history.

**Model results (5-fold temporal cross-validation):**

| Model               | ROC-AUC       | F1-Score      |
|---------------------|---------------|---------------|
| Logistic Regression | 0.674 ± 0.087 | 0.349 ± 0.210 |
| Random Forest       | **0.709 ± 0.055** | 0.378 ± 0.166 |
| XGBoost             | 0.704 ± 0.055 | **0.384 ± 0.151** |
| Neural Network      | 0.709 ± 0.055 | 0.371 ± 0.159 |

Random Forest is the final selected model. XGBoost has slightly better recall if capturing more true positives is the priority.

---

## Running the Notebook

### Option A — Google Colab (recommended, no setup needed)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook** and select `Stats_Mod_Proj.ipynb`
3. Run the first cell — it installs all dependencies automatically:
   ```
   !pip install ucimlrepo plotly scikit-learn xgboost imbalanced-learn -q
   ```
4. When the data-loading cell runs, it will prompt you to upload `bank-additional-full.csv`. Upload the file you downloaded from UCI.
5. Run cells top to bottom. The notebook is divided into:
   - **EDA** (cells 1–23): data exploration, visualizations, insights
   - **Modeling** (cells 24–40): preprocessing, training, evaluation, feature importance

> Note: Plotly charts render interactively in Colab. If you open the notebook in JupyterLab locally, they'll still render — but static matplotlib figures may look slightly different.

---

### Option B — Running Locally

**Prerequisites:** Python 3.9 or higher, pip

**Step 1: Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-optimizer.git
cd bank-marketing-optimizer
```

**Step 2: Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Add the dataset**

Place `bank-additional-full.csv` in the `data/` folder.

Then update the file path in the notebook's data-loading cell from:
```python
file_path = '/content/bank-additional-full.csv'
```
to:
```python
file_path = '../data/bank-additional-full.csv'
```
Also remove or skip the `from google.colab import files` block — that's Colab-specific.

**Step 5: Launch Jupyter**
```bash
jupyter notebook notebook/Stats_Mod_Proj.ipynb
```

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
plotly>=5.14
scikit-learn>=1.3
xgboost>=1.7
imbalanced-learn>=0.11
jupyter>=1.0
```

Or just install from the requirements file:
```bash
pip install -r requirements.txt
```

---

## Methodology Notes

A few design decisions worth knowing if you're replicating or extending this:

**Why temporal cross-validation?**
The data spans 2008–2013, a period where macroeconomic conditions shifted dramatically. Standard k-fold shuffles past and future data together, which leaks future economic conditions into training. `TimeSeriesSplit` (5 folds) trains only on past data and tests on subsequent periods.

**Why downsampling instead of SMOTE?**
The 3:1 majority downsampling ratio follows Moro et al. (2014) directly. SMOTE was considered but introduces synthetic samples in the feature space, which can be misleading when economic indicator features dominate and are highly correlated.

**Why is `duration` excluded?**
Call duration (how long the phone call lasted) is only known after the call ends — by which point you already know whether the customer subscribed or not. Including it trains a model that can't actually be used for pre-call targeting. It's removed in Step 2 of the modeling section.

---

## References

- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- UCI ML Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Author

**Nisarga Patil** — Statistical Methods I, UC Irvine
