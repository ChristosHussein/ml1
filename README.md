# Hands-on AI - Homework 1: E-Commerce Revenue Prediction

## 1. Problem Description
For this project, I chose the E-commerce retail domain. The objective is to predict whether a website visitor will complete a purchase. This is a **classification problem**[cite: 226]. The target variable is `Revenue` (Boolean: True/False)[cite: 227]. Predicting this is highly useful for businesses, as it allows them to identify high-intent shoppers, optimize server resources, or target abandoning users with real-time discounts to save the sale[cite: 227].

## 2. Dataset Description
* **Source:** OpenML "Online Shoppers Intention" dataset[cite: 231].
* **Size:** 12,330 rows and 18 columns, exceeding the minimum requirement of 8,000 rows and 8 columns[cite: 20].
* **Features:** A mix of numerical features (e.g., `Administrative_Duration`, `BounceRates`, `PageValues`) and categorical features (e.g., `VisitorType`, `Month`, `OperatingSystems`)[cite: 26, 231].
* **Target Distribution:** The dataset suffers from a heavy class imbalance. Approximately 85% of sessions ended without a purchase (`False`), while only 15% resulted in revenue (`True`)[cite: 231].

## 3. Preprocessing Approach
To prevent data leakage, the dataset was strictly split (80% Train, 10% Validation, 10% Test) **before** any preprocessing statistics were calculated[cite: 37, 242]. All following transformations were derived exclusively from the training set and subsequently applied to the validation and test sets[cite: 233, 243]:
* **Missing Values:** Handled by computing the median for numerical columns (robust against skewness) and the mode for categorical columns[cite: 59, 60]. 
* **Outliers:** Detected using the Interquartile Range (IQR) method and Winsorized (capped) at the upper and lower bounds to prevent extreme values from distorting the models[cite: 70, 75].
* **Categorical Encoding:** Applied One-Hot Encoding to categorical variables with `drop_first=True` to avoid multicollinearity[cite: 82, 86].
* **Scaling:** Applied `StandardScaler` to normalize the numerical ranges so features with larger magnitudes (like durations) wouldn't dominate the models[cite: 105]. The scaler was saved to `models/scaler.pkl`[cite: 244].

## 4. Feature Engineering
Two new features were derived to encode domain knowledge[cite: 89, 90]:
1. `Total_Duration`: The sum of `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration`. **Intuition:** Captures the total active time a user spent engaged with the website[cite: 234].
2. `Average_Bounce_Exit`: The mean of `BounceRates` and `ExitRates`. **Intuition:** Creates a single, unified metric representing the user's likelihood to abandon the session[cite: 234].

## 5. Exploratory PCA Insights
After scaling, Principal Component Analysis (PCA) was performed[cite: 113].
* **Scree Plot:** The explained variance ratio reveals that it takes several components to capture 90% of the dataset's variance, indicating that shopper behavior is complex and relies on a blend of multiple features[cite: 119, 235]. 
* **Loadings:** Inspecting the components shows that metrics like `PageValues` and our engineered `Total_Duration` strongly dominate the first principal component (PC1)[cite: 120, 235].
* **2D Projection:** The 2D scatter plot projection shows heavy overlapping between the two classes, visually confirming the difficulty of the classification task due to the extreme class imbalance[cite: 121, 236]. 

## 6. Model Comparison
Both models were evaluated on the strictly unseen 10% Test Set[cite: 173]. A baseline Neural Network (2 hidden layers, ReLU, Dropout, Early Stopping) was compared against a Tuned Random Forest (`RandomizedSearchCV` used to tune `n_estimators`, `max_depth`, and `min_samples_split`)[cite: 154, 158, 162, 214].

| Metric | Tuned Random Forest | Neural Network |
| :--- | :--- | :--- |
| **Accuracy** | 83.86% | 85.24% |
| **ROC-AUC** | 0.7941 | 0.7845 |
| **Recall (True Buyers)** | 0.15 | 0.13 |

While the Neural Network achieved a slightly higher raw accuracy, accuracy is heavily misleading on an 85/15 imbalanced dataset[cite: 177, 178]. The Tuned Random Forest achieved a noticeably higher **ROC-AUC score (0.7941)** and successfully recalled more actual buyers (15% vs 13%)[cite: 237]. 

**Surprise Factor:** Given the dataset size, it is not entirely surprising that the tree-based model won. Classical ML algorithms—particularly Random Forests—consistently match or outperform simple deep learning models on tabular, categorical-heavy datasets[cite: 134, 238].

## 7. Best Model Designation
Due to its superior ROC-AUC score and better ability to handle the minority class, the **Tuned Random Forest** is designated as the winning model[cite: 197, 239]. It has been saved as `models/best_model.pkl` to be utilized in future assignments[cite: 198, 199, 244].

## 8. Installation & Execution
Follow these steps to reproduce the pipeline[cite: 240]:

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/ChristosHussein/ml1.git>
   cd hw1