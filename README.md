# Olist Late Delivery Prediction

An end-to-end MLOps project that predicts whether an e-commerce order will be
delivered **late** or **on time**, built the way a real production system
would be rather than as a single notebook. Using the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
the project goes from raw data loaded into a relational MySQL database,
through EDA and feature engineering, to a trained Logistic Regression model
(AUC 0.60), and finally into a validated, logged, and monitored FastAPI
inference service running in Docker — complete with automated testing,
experiment tracking via MLflow, data versioning via DVC, and a CI/CD pipeline
on GitHub Actions. Built independently, with an emphasis on the reasoning
behind each engineering decision (avoiding data leakage, preventing
train-serve skew, documenting real trade-offs and limitations) as much as the
implementation itself.

## Reference Diagrams

**Figure 1: Olist Dataset Table Relationships**

![Image Alt](https://raw.githubusercontent.com/3bdullah-waleed/olist-delivery-prediction/main/image.png)

Shows the 9 raw tables in the dataset and the keys that connect them:
`order_id`, `customer_id`, `product_id`, `seller_id`, and `zip_code_prefix`.
Several of these tables (`order_items`, `order_payments`) have multiple rows
per order, which drove an early design decision: aggregate before joining,
so the final modeling table keeps "one row = one order."

**Figure 2: Order Lifecycle — From Customer to Review**

![Image Alt](https://raw.githubusercontent.com/3bdullah-waleed/olist-delivery-prediction/main/Order%20Lifecycle.png)

Reframes the same schema as a business process: a customer places an order,
the order contains items (products from sellers), the order is paid for, and
after delivery it's reviewed — with customers and sellers both tied back to a
shared geolocation table. This view was the basis for reasoning about data
leakage: reviews and delivery timestamps only exist *after* an order is
placed, so none of that information is available at prediction time and none
of it is used as a model feature.

## What Was Done

**Data & modeling.** All 8 Olist tables were loaded into a normalized MySQL
schema and combined into a single "one row per order" table using SQL
aggregation. Six notebooks (`notebooks/01`–`06`) then take that table through
label creation, a time-based train/val/test split (not random — production
always predicts the future from the past), EDA restricted to the training
split only, feature engineering, and model training. A Logistic Regression
baseline improved from AUC 0.57 to 0.60 after adding two EDA-driven features:
whether the seller and customer are in the same state, and purchase
seasonality (late rates spike noticeably around November and March).

**Production pipeline.** The notebook logic was refactored into reusable
Python modules (`src/`), with the same feature-engineering function used
identically for both training and inference to prevent train-serve skew.
Incoming orders are validated with Great Expectations before reaching the
model, every prediction is logged for later evaluation, and the whole
pipeline is served through a FastAPI service (`/health`, `/predict`,
`/metrics`) that runs in Docker. Data is versioned with DVC, experiments are
tracked and the model is registered in MLflow, 12 automated tests cover
feature logic, data integrity, model behavior, and the API end-to-end, and a
GitHub Actions pipeline runs the test suite and builds the Docker image on
every push — stopping automatically if any test fails.

Known limitations (an MLflow/Docker artifact-serving issue, a local-only DVC
remote, and CD that stops at the build step rather than pushing to a
registry) are documented in detail further down, along with the reasoning
behind each workaround.
