# 📀 Predicting Movie Rental Durations

This project addresses a critical operational challenge for a DVD rental company: **How many days will a customer rent a DVD?** Accurate predictions enable smarter inventory planning, reduce waitlists, and improve customer satisfaction. We treat this as a **supervised regression problem**, experimenting with multiple models (Linear Regression, Ridge, Decision Tree, Random Forest) to predict rental duration in days. The company suggests an MSE ≤ 3 as a benchmark for operational efficiency — but the core objective is to **identify the best-performing model and report its actual test MSE**.

> This project was completed using DataCamp’s Datalab environment.

---

## 🎯 Project Objectives

- Build a **regression model** to predict the number of days a DVD will be rented.
- Perform **feature engineering** by calculating rental duration and creating dummy variables from text features.
- Conduct **feature selection** using Lasso regression to identify the most predictive features.
- **Compare multiple models**: Linear Regression, Ridge, Decision Tree, Random Forest.
- Perform **hyperparameter tuning** for all models (except Linear Regression) using `GridSearchCV`.
- **Preprocess data** by scaling features for models that require it (Linear, Ridge).
- **Identify the best-performing model** based on **lowest Mean Squared Error (MSE)** on the test set.
- Report the **actual MSE** of the winning model — this is the primary deliverable.

> 💡 *Note: The company suggests MSE ≤ 3 as a benchmark for operational efficiency, but model selection is based purely on performance, not this threshold.*

---

## 🗃️ Dataset Overview

The data comes from the `rental_info.csv` file provided by the DVD rental company.

| File | Description |
|------|-------------|
| [`rental_info.csv`](./rental_info.csv) | Rental transactions with customer and movie details |

### Key Columns

| Column | Description |
|--------|-------------|
| `rental_date`, `return_date` | Datetime stamps used to calculate `rental_length_days` (target variable) |
| `amount`, `amount_2` | Rental cost and its square |
| `rental_rate`, `rental_rate_2` | Daily rental rate and its square |
| `release_year`, `length`, `length_2` | Movie metadata |
| `replacement_cost` | Cost to replace the DVD |
| `special_features` | Text field converted into `deleted_scenes` and `behind_the_scenes` dummy variables |
| `NC-17`, `PG`, `PG-13`, `R` | Dummy variables for movie rating (reference category dropped) |

---

## 🔍 Methodology: 5-Step Approach

### 1. Feature Engineering & Preprocessing

Calculated the target variable `rental_length_days` by subtracting `rental_date` from `return_date`. Created new binary features `deleted_scenes` and `behind_the_scenes` from the `special_features` column. Dropped non-numeric columns (`rental_date`, `return_date`, `special_features`) for modeling.

```python
df["rental_length_days"] = (df["return_date"] - df["rental_date"]).dt.days
df["deleted_scenes"] = np.where(df["special_features"].str.contains("Deleted Scenes"), 1, 0)
```

### 2. Train-Test Split and Scaling

Split the data 80/20. Scaled features using `StandardScaler` for Linear and Ridge regression models.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Feature Selection with Lasso

Used Lasso regression (alpha=0.1) to identify the most important features. A bar plot of coefficients suggested `amount` was the strongest predictor.

### 4. Model Training, Tuning, and Evaluation

Defined a function `FineTune_predict_eval` to handle hyperparameter tuning with `GridSearchCV` and evaluation. Tested four models:
- **LinearRegression**: No hyperparameters, used as a baseline.
- **Ridge**: Tuned `alpha`.
- **DecisionTreeRegressor**: Tuned `max_depth`, `min_samples_split`, `min_samples_leaf`.
- **RandomForestRegressor**: Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

Each model was evaluated by its **Root Mean Squared Error (RMSE)** on the test set, then converted to MSE.

### 5. Selecting the Best Model

The model with the lowest RMSE (and thus lowest MSE) was selected. Its **actual MSE** is the final output — regardless of whether it meets the suggested benchmark of 3.

```python
best_model_name = min(results, key=lambda resName: results[resName][1])
best_model, best_rmse = results[best_model_name]
best_mse = best_rmse ** 2  # This is the key deliverable
```

---

## 📊 Key Findings

- ✅ **Best Model:** Selected programmatically as the one with the lowest RMSE → converted to **actual MSE**. (Likely Random Forest or Decision Tree due to ability to capture non-linear patterns.)
- 📌 **Performance Metric:** The final variable `best_mse` holds the **true test MSE of the winning model** — this is the project’s core output.
- 🧪 **Feature Insight:** Lasso regression indicated that `amount` (the rental cost) is the strongest single predictor of rental duration.
- ✅ The pipeline delivers a **robust, automated model comparison framework** — ideal for real-world deployment where model performance must be empirically determined.

> 💡 Final Output:
> ```python
> best_model_name  # "R_forest"
> best_mse         # 1.9775150039532432

> ```

---

## 🛠️ Tools Used

- **Python**
- **pandas, numpy** – for data manipulation and feature engineering
- **scikit-learn** – `train_test_split`, `StandardScaler`, `Lasso`, `LinearRegression`, `Ridge`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GridSearchCV`, `mean_squared_error`
- **Matplotlib** – for visualizing Lasso coefficients
- **Jupyter Notebook / DataCamp Datalab** – for analysis and reporting

---

## 📌 How to Use

This project was completed in **DataCamp’s Datalab environment**. To reproduce:

1. Upload `rental_info.csv` to your workspace.
2. Open the notebook.
3. Run all cells to execute the full pipeline: from feature engineering to model selection and final MSE calculation.

> 🔗 This project was created by **DataCamp** as part of their machine learning curriculum. You can find the original exercise on their platform.

---

## ✍️ Author

Completed by **Achraf Salimi** — building predictive models to solve real-world business problems in inventory and operations.
