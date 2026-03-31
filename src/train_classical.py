import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Task 3.1 & Task 6: Trains and TUNES a Random Forest using RandomizedSearchCV.
    """
    print("\n--- Task 6: Tuning Classical ML Model (Random Forest) ---")
    
    # 1. Define the grid of hyperparameters to search
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # 2. Initialize the base model and the search object
    base_rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    random_search = RandomizedSearchCV(
        base_rf, 
        param_distributions=param_dist, 
        n_iter=10, # Tests 10 random combinations
        cv=3,      # 3-Fold Cross Validation
        random_state=42, 
        n_jobs=-1  # Uses all CPU cores
    )
    
    # 3. Fit the search on the Training set
    print("Searching for the best hyperparameters... (This may take a minute)")
    random_search.fit(X_train, y_train)
    
    # 4. Extract the best model
    best_model = random_search.best_estimator_
    print(f"Best Parameters Found: {random_search.best_params_}")
    
    # 5. Evaluate the BEST model on the Validation Set
    predictions = best_model.predict(X_val)
    probabilities = best_model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, predictions)
    roc_auc = roc_auc_score(y_val, probabilities)
    
    print(f"Tuned Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Tuned Validation ROC-AUC:  {roc_auc:.4f}")
    
    # Save the tuned model
    joblib.dump(best_model, 'models/classical_model.pkl')
    
    return best_model

def test_model(model, X_test, y_test):
    """Task 4: Evaluates the trained model on the final, unseen test set."""
    print("\n--- Task 4: Final Test Set Evaluation (Tuned Classical Model) ---")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Test ROC-AUC:  {roc_auc:.4f}")
    print("\nFinal Test Classification Report:")
    print(classification_report(y_test, predictions))