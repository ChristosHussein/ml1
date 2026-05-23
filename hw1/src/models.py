from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Trains a Random Forest model and evaluates it on the validation set.
    """
    print("\n--- Training Random Forest Model ---")
    
    # 1. Initialize the model (random_state=42 keeps your results reproducible)
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    
    # 2. Train the model using the 80% Training Data
    model.fit(X_train, y_train)
    
    # 3. Ask the model to make predictions on the 10% Validation Data
    predictions = model.predict(X_val)
    
    # 4. Grade the model's homework
    accuracy = accuracy_score(y_val, predictions)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    # This prints out Precision, Recall, and F1-Score for both classes (Buyers and Non-Buyers)
    print(classification_report(y_val, predictions))
    
    return model

def test_model(model, X_test, y_test):
    """
    Evaluates the trained model on the final, unseen test set.
    """
    print("\n--- Final Test Set Evaluation ---")
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print("\nFinal Test Classification Report:")
    print(classification_report(y_test, predictions))