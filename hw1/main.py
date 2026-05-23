from src.preprocessing import (
    load_and_split_data, 
    handle_missing_values, 
    treat_outliers_iqr, 
    encode_categorical, 
    engineer_features, 
    scale_features
)
from src.evaluate import run_exploratory_pca
from src.train_classical import train_and_evaluate_model, test_model
from src.train_neural import train_and_evaluate_nn, test_nn_model

def main():
    print("Starting Machine Learning Pipeline...")
    
    # Task 2.1: Load and Split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        filepath="data/dataset.csv", 
        target_column="Revenue"
    )

    # Task 2.2: Handle Missing Values
    X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test)

    # Task 2.3: Treat Outliers
    X_train, X_val, X_test = treat_outliers_iqr(X_train, X_val, X_test)

    # Task 2.4: Encode Categorical Data
    X_train, X_val, X_test = encode_categorical(X_train, X_val, X_test)

    # Task 2.5: Feature Engineering
    X_train, X_val, X_test = engineer_features(X_train, X_val, X_test)

    # Task 2.6: Scale Features (and save the scaler!)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    
    # Task 2.7: Run Exploratory PCA
    run_exploratory_pca(X_train, y_train)
    
    print("\n--- Preprocessing Complete! ---")
    
    # Task 3.1: Train Classical Model (Random Forest)
    rf_model = train_and_evaluate_model(X_train, y_train, X_val, y_val)
    
    # Task 3.2: Train Neural Network
    nn_model = train_and_evaluate_nn(X_train, y_train, X_val, y_val)
    
    # Task 4: Final Evaluation on the locked Test Set
    test_model(rf_model, X_test, y_test)
    test_nn_model(nn_model, X_test, y_test)

if __name__ == "__main__":
    main()