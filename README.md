E-commerce Customer Analytics

Data analytics project focused on understanding customer behavior, identifying valuable customers, predicting churn, and uncovering the products that drive revenue for an online retail business.


📌 Business Scenario

An online retailer has thousands of customers but doesn't know:

Who the valuable customers are
Who is about to stop buying (churn risk)
Which products drive revenue

As a Data Analyst, the goal of this project is to improve customer retention and increase revenue through data-driven insights.

📂 Dataset

The project uses the following relational datasets:

Table	Description
customers	Customer profile and demographic details
orders	Order-level transaction data
order_items	Line-item details for each order
products	Product catalog and category information
payments	Payment method and transaction status
reviews	Customer product reviews and ratings

Update this section with actual source/link of the dataset (e.g., Kaggle, company database, etc.)

❓ Business Questions
💰 Revenue
Monthly sales trend
Year-over-year (YoY) growth
Average Order Value (AOV)
👥 Customers
Repeat purchase rate
Customer Lifetime Value (CLV)
New vs Returning customers
Top 10 customers by revenue
📦 Products
Best-selling categories
Low-performing products
Most profitable products
🧩 Customer Segmentation — RFM Analysis

Customers are segmented using RFM Analysis:

Metric	Meaning
Recency (R)	How recently did the customer purchase?
Frequency (F)	How often do they purchase?
Monetary (M)	How much have they spent?
Customer Segments
🏆 Champions
💎 Loyal Customers
🌱 Potential Loyalists
⚠️ At Risk
❌ Lost Customers
🐍 Python Analysis

The following analyses are performed using Python:

RFM Scoring – Quantile-based scoring of Recency, Frequency, and Monetary values
K-Means Clustering – Unsupervised clustering to validate/refine RFM segments
Cohort Analysis – Tracking customer retention over time by acquisition month
Customer Retention Analysis – Measuring repeat purchase and churn behavior
Tech Stack
pandas, numpy – Data wrangling
matplotlib, seaborn – Visualization
scikit-learn – K-Means clustering
jupyter notebook – Analysis environment
📊 Dashboard

Interactive dashboard(s) covering:

Revenue Dashboard – Sales trends, growth, AOV
Customer Segments Dashboard – RFM segment distribution and value
Product Performance Dashboard – Category and product-level insights
Retention Dashboard – Cohort retention curves and churn trends

Add your dashboard tool here (Power BI / Tableau / Looker Studio) and a screenshot or live link once built.

💡 Business Recommendation

A small percentage of customers generate a large share of revenue (Pareto principle). Personalized loyalty programs targeting these high-value ("Champion" and "Loyal") customers could significantly improve retention and long-term revenue.

📁 Project Structure
ecommerce-customer-analytics/
│
├── data/                   # Raw and processed datasets
├── notebooks/               # Jupyter notebooks (RFM, clustering, cohort analysis)
├── dashboard/                # Dashboard files/screenshots
├── src/                     # Python scripts (data cleaning, RFM, clustering)
├── outputs/                  # Reports, visualizations, exported results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
⚙️ Setup & Installation
Clone the repository
bash
git clone https://github.com/<your-username>/ecommerce-customer-analytics.git
cd ecommerce-customer-analytics
Create a virtual environment
bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies
bash
pip install -r requirements.txt
Run the analysis
bash
jupyter notebook notebooks/rfm_analysis.ipynb
🚀 Usage
Place raw datasets inside the data/ folder
Run notebooks in notebooks/ in order:
Data cleaning & preprocessing
RFM scoring
K-Means clustering
Cohort & retention analysis
Generated outputs (charts, tables) will be saved in outputs/
Load dashboard files from dashboard/ into your BI tool of choice
📈 Key Insights (Sample)

Update this section once analysis is complete

X% of customers are classified as "Champions" and contribute Y% of total revenue
Average customer retention rate is Z% month-over-month
Top category "___" drives the highest revenue share
🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

📄 License

This project is licensed under the MIT License.

📬 Contact

Author: Your Name Email: your.email@example.com LinkedIn: your-linkedin

Appendix A: Python Source Code (rfm_analysis.py)
python
"""
E-commerce Customer Analytics
=============================
Performs:
  1. Data loading & merging (customers, orders, order_items, products, payments, reviews)
  2. RFM Scoring (Recency, Frequency, Monetary)
  3. Customer Segmentation (Champions, Loyal, Potential Loyalists, At Risk, Lost)
  4. K-Means Clustering (validate/refine RFM segments)
  5. Cohort Analysis (monthly retention by acquisition cohort)
  6. Customer Retention Analysis (repeat purchase rate, churn)

Expected input CSVs (place inside ./data/):
  - customers.csv     : customer_id, signup_date, ...
  - orders.csv        : order_id, customer_id, order_date, order_status, ...
  - order_items.csv   : order_id, product_id, quantity, price
  - products.csv      : product_id, category, ...
  - payments.csv       : order_id, payment_value, payment_type
  - reviews.csv        : order_id, review_score

Output:
  - outputs/rfm_scores.csv
  - outputs/rfm_segments.csv
  - outputs/kmeans_clusters.csv
  - outputs/cohort_retention.csv
  - outputs/summary_report.txt
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data():
    """Load and merge raw datasets into a single transaction-level dataframe."""
    customers = pd.read_csv(f"{DATA_DIR}/customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(f"{DATA_DIR}/order_items.csv")
    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    payments = pd.read_csv(f"{DATA_DIR}/payments.csv")

    # Merge order_items with orders, products, payments
    df = order_items.merge(orders, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(
        payments.groupby("order_id")["payment_value"].sum().reset_index(),
        on="order_id", how="left"
    )
    df = df.merge(customers, on="customer_id", how="left")

    # Line total (fallback to quantity * price if payment_value missing)
    df["line_total"] = df["quantity"] * df["price"]

    return df, customers, orders


# ---------------------------------------------------------------------------
# 2. RFM SCORING
# ---------------------------------------------------------------------------
def compute_rfm(df, snapshot_date=None):
    """Compute Recency, Frequency, Monetary values per customer and score them 1-5."""
    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("line_total", "sum")
    ).reset_index()

    # Score 1-5 using quantiles (5 = best)
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    return rfm


def segment_customers(rfm):
    """Classify customers into segments based on RFM scores."""
    def classify(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Lost Customers"
        else:
            return "Others"

    rfm["segment"] = rfm.apply(classify, axis=1)
    return rfm


# ---------------------------------------------------------------------------
# 3. K-MEANS CLUSTERING
# ---------------------------------------------------------------------------
def kmeans_clustering(rfm, n_clusters=5):
    """Cluster customers using K-Means on scaled RFM values."""
    features = rfm[["recency", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    return rfm, kmeans


# ---------------------------------------------------------------------------
# 4. COHORT ANALYSIS
# ---------------------------------------------------------------------------
def cohort_analysis(orders):
    """Build a monthly cohort retention table based on customer's first order month."""
    orders = orders.copy()
    orders["order_month"] = orders["order_date"].dt.to_period("M")

    first_purchase = orders.groupby("customer_id")["order_month"].min().reset_index()
    first_purchase.columns = ["customer_id", "cohort_month"]

    orders = orders.merge(first_purchase, on="customer_id", how="left")
    orders["cohort_index"] = (
        (orders["order_month"].dt.year - orders["cohort_month"].dt.year) * 12
        + (orders["order_month"].dt.month - orders["cohort_month"].dt.month)
    )

    cohort_data = orders.groupby(["cohort_month", "cohort_index"])["customer_id"] \
        .nunique().reset_index()

    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0).round(3) * 100

    return retention


# ---------------------------------------------------------------------------
# 5. RETENTION METRICS
# ---------------------------------------------------------------------------
def retention_metrics(orders):
    """Compute repeat purchase rate and new vs returning customer counts."""
    order_counts = orders.groupby("customer_id")["order_id"].nunique()
    repeat_customers = (order_counts > 1).sum()
    total_customers = order_counts.shape[0]
    repeat_purchase_rate = round(repeat_customers / total_customers * 100, 2)

    return {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_purchase_rate_%": repeat_purchase_rate
    }


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df, customers, orders = load_data()

    print("Computing RFM scores...")
    rfm = compute_rfm(df)
    rfm = segment_customers(rfm)
    rfm.to_csv(f"{OUTPUT_DIR}/rfm_segments.csv", index=False)

    print("Running K-Means clustering...")
    rfm, model = kmeans_clustering(rfm)
    rfm.to_csv(f"{OUTPUT_DIR}/kmeans_clusters.csv", index=False)

    print("Building cohort retention table...")
    retention_table = cohort_analysis(orders)
    retention_table.to_csv(f"{OUTPUT_DIR}/cohort_retention.csv")

    print("Calculating retention metrics...")
    metrics = retention_metrics(orders)

    # Summary report
    with open(f"{OUTPUT_DIR}/summary_report.txt", "w") as f:
        f.write("E-COMMERCE CUSTOMER ANALYTICS - SUMMARY\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total Customers: {metrics['total_customers']}\n")
        f.write(f"Repeat Customers: {metrics['repeat_customers']}\n")
        f.write(f"Repeat Purchase Rate: {metrics['repeat_purchase_rate_%']}%\n\n")
        f.write("Segment Distribution:\n")
        f.write(rfm["segment"].value_counts().to_string())
        f.write("\n\nTop 10 Customers by Monetary Value:\n")
        f.write(rfm.sort_values("monetary", ascending=False).head(10)[
            ["customer_id", "recency", "frequency", "monetary", "segment"]
        ].to_string(index=False))

    print(f"Done. Outputs saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
Appendix B: SQL Business Queries (business_queries.sql)
sql
-- ============================================================
-- E-commerce Customer Analytics - SQL Queries
-- ============================================================
-- This file contains SQL queries to answer the core business
-- questions defined in the project (Revenue, Customers, Products,
-- RFM Segmentation).
--
-- Tables used:
--   customers(customer_id, signup_date, ...)
--   orders(order_id, customer_id, order_date, order_status, ...)
--   order_items(order_id, product_id, quantity, price)
--   products(product_id, category, ...)
--   payments(order_id, payment_value, payment_type)
--   reviews(order_id, review_score)
-- ============================================================


-- ------------------------------------------------------------
-- 1. REVENUE
-- ------------------------------------------------------------

-- 1.1 Monthly Sales
-- Total revenue generated per month
SELECT
    DATE_TRUNC('month', o.order_date) AS month,
    SUM(oi.quantity * oi.price) AS monthly_revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY 1
ORDER BY 1;


-- 1.2 Year-over-Year (YoY) Growth
-- Compares each year's revenue to the previous year
WITH yearly_revenue AS (
    SELECT
        EXTRACT(YEAR FROM o.order_date) AS year,
        SUM(oi.quantity * oi.price) AS revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY 1
)
SELECT
    year,
    revenue,
    LAG(revenue) OVER (ORDER BY year) AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY year))
        / LAG(revenue) OVER (ORDER BY year) * 100, 2
    ) AS yoy_growth_percent
FROM yearly_revenue
ORDER BY year;


-- 1.3 Average Order Value (AOV)
-- Total revenue divided by total number of orders
SELECT
    ROUND(SUM(oi.quantity * oi.price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id;


-- ------------------------------------------------------------
-- 2. CUSTOMERS
-- ------------------------------------------------------------

-- 2.1 Repeat Purchase Rate
-- % of customers who placed more than one order
SELECT
    ROUND(
        SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS repeat_purchase_rate_percent
FROM (
    SELECT customer_id, COUNT(DISTINCT order_id) AS order_count
    FROM orders
    GROUP BY customer_id
) t;


-- 2.2 Customer Lifetime Value (CLV)
-- Total revenue generated by each customer (simple historical CLV)
SELECT
    c.customer_id,
    SUM(oi.quantity * oi.price) AS lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id
ORDER BY lifetime_value DESC;


-- 2.3 New vs Returning Customers (per month)
-- Classifies each customer's orders as "New" (first order) or "Returning"
WITH first_orders AS (
    SELECT
        customer_id,
        MIN(order_date) AS first_order_date
    FROM orders
    GROUP BY customer_id
)
SELECT
    DATE_TRUNC('month', o.order_date) AS month,
    CASE
        WHEN DATE_TRUNC('month', o.order_date) = DATE_TRUNC('month', f.first_order_date)
        THEN 'New'
        ELSE 'Returning'
    END AS customer_type,
    COUNT(DISTINCT o.customer_id) AS customer_count
FROM orders o
JOIN first_orders f ON o.customer_id = f.customer_id
GROUP BY 1, 2
ORDER BY 1, 2;


-- 2.4 Top 10 Customers by Revenue
SELECT
    c.customer_id,
    SUM(oi.quantity * oi.price) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 3. PRODUCTS
-- ------------------------------------------------------------

-- 3.1 Best-Selling Categories (by revenue)
SELECT
    p.category,
    SUM(oi.quantity * oi.price) AS category_revenue,
    SUM(oi.quantity) AS units_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY category_revenue DESC;


-- 3.2 Low-Performing Products (bottom 10 by revenue)
SELECT
    p.product_id,
    p.category,
    SUM(oi.quantity * oi.price) AS product_revenue,
    SUM(oi.quantity) AS units_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_id, p.category
ORDER BY product_revenue ASC
LIMIT 10;


-- 3.3 Most Profitable Products (top 10 by revenue)
SELECT
    p.product_id,
    p.category,
    SUM(oi.quantity * oi.price) AS product_revenue,
    SUM(oi.quantity) AS units_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_id, p.category
ORDER BY product_revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 4. RFM ANALYSIS (Recency, Frequency, Monetary)
-- ------------------------------------------------------------

-- 4.1 Raw RFM values per customer
-- Recency  = days since last order (relative to most recent order date in data)
-- Frequency = number of distinct orders
-- Monetary  = total amount spent
WITH rfm_raw AS (
    SELECT
        o.customer_id,
        DATE_PART('day', (SELECT MAX(order_date) FROM orders) - MAX(o.order_date)) AS recency,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.quantity * oi.price) AS monetary
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.customer_id
)
SELECT * FROM rfm_raw
ORDER BY monetary DESC;


-- 4.2 RFM Scoring (1-5 using NTILE quantiles)
-- Higher score = better (5 = best recency/frequency/monetary)
WITH rfm_raw AS (
    SELECT
        o.customer_id,
        DATE_PART('day', (SELECT MAX(order_date) FROM orders) - MAX(o.order_date)) AS recency,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.quantity * oi.price) AS monetary
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.customer_id
)
SELECT
    customer_id,
    recency,
    frequency,
    monetary,
    -- Recency: lower days = higher score, so reverse the NTILE order
    (6 - NTILE(5) OVER (ORDER BY recency)) AS r_score,
    NTILE(5) OVER (ORDER BY frequency)     AS f_score,
    NTILE(5) OVER (ORDER BY monetary)      AS m_score
FROM rfm_raw
ORDER BY customer_id;


-- 4.3 Customer Segmentation based on RFM scores
-- Wraps the scoring query above and classifies customers into segments
WITH rfm_raw AS (
    SELECT
        o.customer_id,
        DATE_PART('day', (SELECT MAX(order_date) FROM orders) - MAX(o.order_date)) AS recency,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.quantity * oi.price) AS monetary
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.customer_id
),
rfm_scored AS (
    SELECT
        customer_id,
        (6 - NTILE(5) OVER (ORDER BY recency)) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary)      AS m_score
    FROM rfm_raw
)
SELECT
    customer_id,
    r_score, f_score, m_score,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'Potential Loyalists'
        WHEN r_score <= 2 AND f_score >= 3                  THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2                  THEN 'Lost Customers'
        ELSE 'Others'
    END AS segment
FROM rfm_scored
ORDER BY customer_id;
