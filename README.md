# 🛒 E-commerce Customer Analytics

Data analytics project focused on understanding customer behavior, identifying valuable customers, predicting churn, and uncovering the products that drive revenue for an online retail business.

!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)
!\[Status](https://img.shields.io/badge/status-active-brightgreen)
!\[License](https://img.shields.io/badge/license-MIT-lightgrey)

\---

## 📌 Business Scenario

An online retailer has thousands of customers but doesn't know:

* Who the **valuable customers** are
* Who is **about to stop buying** (churn risk)
* Which **products drive revenue**

As a Data Analyst, the goal of this project is to **improve customer retention** and **increase revenue** through data-driven insights.

\---

## 📂 Dataset

The project uses the following relational datasets:

|Table|Description|
|-|-|
|`customers`|Customer profile and demographic details|
|`orders`|Order-level transaction data|
|`order\\\\\\\_items`|Line-item details for each order|
|`products`|Product catalog and category information|
|`payments`|Payment method and transaction status|
|`reviews`|Customer product reviews and ratings|

> Update this section with actual source/link of the dataset (e.g., Kaggle, company database, etc.)

\---

## ❓ Business Questions

### 💰 Revenue

* Monthly sales trend
* Year-over-year (YoY) growth
* Average Order Value (AOV)

### 👥 Customers

* Repeat purchase rate
* Customer Lifetime Value (CLV)
* New vs Returning customers
* Top 10 customers by revenue

### 📦 Products

* Best-selling categories
* Low-performing products
* Most profitable products

\---

## 🧩 Customer Segmentation — RFM Analysis

Customers are segmented using **RFM Analysis**:

|Metric|Meaning|
|-|-|
|**Recency (R)**|How recently did the customer purchase?|
|**Frequency (F)**|How often do they purchase?|
|**Monetary (M)**|How much have they spent?|

### Customer Segments

* 🏆 **Champions**
* 💎 **Loyal Customers**
* 🌱 **Potential Loyalists**
* ⚠️ **At Risk**
* ❌ **Lost Customers**

\---

## 🐍 Python Analysis

The following analyses are performed using Python:

* **RFM Scoring** – Quantile-based scoring of Recency, Frequency, and Monetary values
* **K-Means Clustering** – Unsupervised clustering to validate/refine RFM segments
* **Cohort Analysis** – Tracking customer retention over time by acquisition month
* **Customer Retention Analysis** – Measuring repeat purchase and churn behavior

### Tech Stack

* `pandas`, `numpy` – Data wrangling
* `matplotlib`, `seaborn` – Visualization
* `scikit-learn` – K-Means clustering
* `jupyter notebook` – Analysis environment

\---

## 📊 Dashboard

Interactive dashboard(s) covering:

1. **Revenue Dashboard** – Sales trends, growth, AOV
2. **Customer Segments Dashboard** – RFM segment distribution and value
3. **Product Performance Dashboard** – Category and product-level insights
4. **Retention Dashboard** – Cohort retention curves and churn trends

> Add your dashboard tool here (Power BI / Tableau / Looker Studio) and a screenshot or live link once built.

\---

## 💡 Business Recommendation

A small percentage of customers generate a large share of revenue (Pareto principle). **Personalized loyalty programs** targeting these high-value ("Champion" and "Loyal") customers could significantly improve retention and long-term revenue.

\---

## 📁 Project Structure

```
ecommerce-customer-analytics/
│
├── data/                   # Raw and processed datasets
├── notebooks/               # Jupyter notebooks (RFM, clustering, cohort analysis)
├── dashboard/                # Dashboard files/screenshots
├── src/                     # Python scripts (data cleaning, RFM, clustering)
├── outputs/                  # Reports, visualizations, exported results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

\---

## ⚙️ Setup \& Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/ecommerce-customer-analytics.git
cd ecommerce-customer-analytics
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\\\\\\\Scripts\\\\\\\\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the analysis**

```bash
jupyter notebook notebooks/rfm\\\\\\\_analysis.ipynb
```

\---

## 🚀 Usage

* Place raw datasets inside the `data/` folder
* Run notebooks in `notebooks/` in order:

  1. Data cleaning \& preprocessing
  2. RFM scoring
  3. K-Means clustering
  4. Cohort \& retention analysis
* Generated outputs (charts, tables) will be saved in `outputs/`
* Load dashboard files from `dashboard/` into your BI tool of choice

\---

## 📈 Key Insights (Sample)

> Update this section once analysis is complete

* X% of customers are classified as "Champions" and contribute Y% of total revenue
* Average customer retention rate is Z% month-over-month
* Top category "\_\_\_" drives the highest revenue share

\---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

\---

## 📄 License

This project is licensed under the MIT License.

\---

## 📬 Contact

**Author:** velan k
**Email:** velan9685@gmail.com


