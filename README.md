# Hands-on AI: Homework 1 - E-commerce Machine Learning Pipeline

## 1. Problem Description
I chose the E-commerce domain to predict online shopper purchasing intention. [cite_start]This is a classification problem[cite: 226]. [cite_start]The target variable is `Revenue` (whether a customer made a purchase or not), which is highly useful for businesses to identify potential buyers and target them with promotions[cite: 227].

## 2. Dataset Description
* [cite_start]**Source:** OpenML "Online Shoppers Intention" dataset[cite: 231].
* [cite_start]**Size:** Over 12,000 rows and 17 columns (exceeding the 8,000 row/8 column requirement)[cite: 231].
* [cite_start]**Distribution:** Heavy class imbalance (approx. 85% False / 15% True for the target variable)[cite: 231].

## 3. Preprocessing Approach
* [cite_start]**Split First:** The data was split 80/10/10 using stratified splitting before any statistics were computed[cite: 233].
* [cite_start]**Missing Values:** Numerical columns were filled with the median, categorical with the mode[cite: 232].
* [cite_start]**Outliers:** Handled using the IQR method (Winsorizing/capping)[cite: 232].
* [cite_start]**Encoding:** One-hot encoding was used for categorical features like Month and VisitorType[cite: 232].
* [cite_start]**Scaling:** `StandardScaler` was fitted on the training data only, then applied to the validation and test sets[cite: 232].

## 4. Feature Engineering
[cite_start]Created two new features[cite: 234]:
1. `Total_Duration`: Sum of Administrative, Informational, and ProductRelated duration (intuition: overall time engaged).
2. `Average_Bounce_Exit`: Average of BounceRates and ExitRates (intuition: combined metric of likelihood to leave the site).

## 5. PCA Insights
[cite_start]*(Open your `pca_scree_plot.png` and `pca_2d_scatter.png` images to write this part! Mention how many components it takes to reach high variance, and what the 2D plot looks like)*[cite: 235, 236].

## 6. Model Comparison
* **Random Forest:** Accuracy 83.62%, ROC-AUC 0.7871
* **Neural Network:** Accuracy 84.83%, ROC-AUC 0.7903
[cite_start]The Neural Network slightly outperformed the Random Forest[cite: 237]. Both models struggled with minority class recall due to the heavy 85/15 class imbalance. [cite_start]It is slightly surprising that the NN won, as tree-based models typically excel on small tabular datasets[cite: 238].

## 7. Best Model Designation
[cite_start]Because it achieved a higher Accuracy and ROC-AUC score on the strict Test set, the Neural Network was designated as the best model and saved as `models/best_model.h5`[cite: 239].

## 8. Installation & Execution
```bash
git clone <your-github-repo-url>
cd hw1
python -m pip install -r requirements.txt
python main.py