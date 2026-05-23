# Hands-on AI: Homework 1 - E-commerce Machine Learning Pipeline

## 1. Problem Description
For this project, I chose the **E-commerce** domain. This is a **classification problem** where the goal is to predict the target variable `Revenue` (True if a purchase was made, False otherwise). This task is highly valuable for online retailers as it allows them to identify high-intent visitors in real-time, optimize marketing spend, and offer targeted incentives to users likely to abandon their carts.

## 2. Dataset Description
* **Source:** [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
* **Reference:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Comput & Applic 31, 6893–6908 (2019).
* **Size:** 12,330 rows and 18 columns (Exceeds the 8k/8 requirement).
* **Target Distribution:** The dataset is heavily imbalanced, with approximately 84.5% of sessions resulting in no purchase (`False`) and 15.5% resulting in a purchase (`True`).

## 3. Preprocessing Approach
To ensure a rigorous evaluation and prevent data leakage, the data was split (80% Train, 10% Val, 10% Test) **before** any transformations.
* **Missing Values:** Numerical features were imputed with the **median** and categorical features with the **mode**, calculated strictly from the training set.
* **Outliers:** Handled via the **IQR method**, where values outside 1.5 * IQR were capped (Winsorized) to reduce the impact of extreme variance.
* **Encoding:** Categorical variables were transformed using **One-Hot Encoding** with `drop_first=True`.
* **Scaling:** Features were normalized using `StandardScaler` (fitted on training data only) to ensure all features contribute equally to the model.

## 4. Feature Engineering
Two new features were engineered to capture user engagement:
1. `Total_Duration`: Sum of Administrative, Informational, and ProductRelated durations. **Intuition:** More time spent actively on site generally correlates with higher purchase intent.
2. `Average_Bounce_Exit`: The mean of Bounce and Exit rates. **Intuition:** A unified metric representing the "friction" or likelihood of a user leaving without converting.

## 5. PCA Insights
* **Scree Plot:** Reveals that a high number of components are required to explain 90% of the variance, suggesting that shopper intent is multidimensional and not driven by a single "silver bullet" feature.
* **Loadings:** Features like `PageValues` and `ExitRates` showed the highest influence on the first two principal components.
* **2D Projection:** The 2D visualization shows significant overlap between classes, highlighting the complexity of the classification task given the class imbalance.

## 6. Model Comparison
| Metric | Tuned Random Forest | Neural Network |
| **Test Accuracy** | 83.86%   |     85.24%     |
| **Test ROC-AUC** | **0.7941**|     0.7845     |

**Outcome:** While the Neural Network had slightly higher raw accuracy, the **Tuned Random Forest** achieved a higher **ROC-AUC (0.7941)**. In imbalanced datasets, ROC-AUC is a more reliable metric. This outcome is consistent with industry experience where tree-based models often outperform deep learning on smaller, tabular datasets.

## 7. Best Model Designation
The **Tuned Random Forest** was designated as `best_model.pkl`. It was selected because it provided the best balance between precision and recall for the minority class (purchasers) and achieved the highest ROC-AUC score on the final test set.

## 8. Installation & Execution
Follow these steps to run the pipeline:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChristosHussein/ml1.git
   cd ml1