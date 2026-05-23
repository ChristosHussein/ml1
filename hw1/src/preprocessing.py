import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_split_data(filepath, target_column):
    """
    Task 2.1: Loads the dataset and splits it into 80% Train, 10% Validation, and 10% Test.
    """
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print("Task 2.1: Data split successful!")
    print(f"Train: {X_train.shape[0]} rows | Validation: {X_val.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_missing_values(X_train, X_val, X_test):
    """
    Task 2.2: Fills missing values using statistics calculated ONLY from the training set.
    """
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns

    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col] = X_val[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    print("Task 2.2: Missing values handled successfully!")
    return X_train, X_val, X_test

def treat_outliers_iqr(X_train, X_val, X_test):
    """
    Task 2.3: Detects outliers using IQR on the training set and caps them across all sets.
    """
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    
    for col in numerical_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
        X_val[col] = X_val[col].clip(lower=lower_bound, upper=upper_bound)
        X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)
        
    print("Task 2.3: Outliers treated (capped) successfully!")
    return X_train, X_val, X_test

def encode_categorical(X_train, X_val, X_test):
    """
    Task 2.4: Converts text columns into 1s and 0s using One-Hot Encoding.
    """
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns
    
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print("Task 2.4: Categorical data encoded successfully!")
    return X_train, X_val, X_test

def engineer_features(X_train, X_val, X_test):
    """
    Task 2.5: Creates new features from existing columns.
    """
    datasets = [X_train, X_val, X_test]
    for df in datasets:
        # Feature 1: Total time spent on the website
        if all(col in df.columns for col in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']):
            df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
            
        # Feature 2: Combined exit/bounce metric
        if all(col in df.columns for col in ['BounceRates', 'ExitRates']):
            df['Average_Bounce_Exit'] = (df['BounceRates'] + df['ExitRates']) / 2
            
    print("Task 2.5: Feature engineering complete! 2 new features added.")
    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    """
    Task 2.6: Scales numerical features and saves the scaler to disk.
    """
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    
    # Calculate the rules ONLY on the training set, and transform it
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Apply the exact same rules to the Validation and Test sets
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save the fitted scaler for Homework 2
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Task 2.6: Features scaled successfully! Scaler saved to models/scaler.pkl")
    return X_train, X_val, X_test