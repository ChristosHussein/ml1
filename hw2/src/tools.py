import joblib
import pandas as pd
from langchain_core.tools import tool

# Φορτώνουμε τα μοντέλα από το HW1
try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("HW1 Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load HW1 models. Error: {e}")

@tool
def predict_purchase(
    administrative: int, administrative_duration: float, informational: int, informational_duration: float,
    product_related: int, product_related_duration: float, bounce_rates: float, exit_rates: float,
    page_values: float, special_day: float, month: str, operating_systems: int, browser: int,
    region: int, traffic_type: int, visitor_type: str, weekend: bool
) -> str:
    """
    Predicts whether an e-commerce visitor will make a purchase.
    Provide all numerical and categorical features of the session to get a prediction.
    """
    # 1. Αρχικά δεδομένα από τον χρήστη
    input_dict = {
        "Administrative": administrative, "Administrative_Duration": administrative_duration,
        "Informational": informational, "Informational_Duration": informational_duration,
        "ProductRelated": product_related, "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates, "ExitRates": exit_rates,
        "PageValues": page_values, "SpecialDay": special_day,
        "Month": month, "OperatingSystems": operating_systems,
        "Browser": browser, "Region": region,
        "TrafficType": traffic_type, "VisitorType": visitor_type,
        "Weekend": weekend
    }
    input_data = pd.DataFrame([input_dict])

    # 2. Feature Engineering του HW1
    input_data['Total_Duration'] = (input_data['Administrative_Duration'] + 
                                    input_data['Informational_Duration'] + 
                                    input_data['ProductRelated_Duration'])
    input_data['Average_Bounce_Exit'] = (input_data['BounceRates'] + input_data['ExitRates']) / 2

    # 3. Dummy Variables
    input_data = pd.get_dummies(input_data)

    # 4. ΕΥΘΥΓΡΑΜΜΙΣΗ ΜΕ ΤΟΝ SCALER
    scaler_cols = getattr(scaler, 'feature_names_in_', None)
    if scaler_cols is not None:
        # Δίνουμε στον scaler ΑΚΡΙΒΩΣ τις στήλες που περιμένει
        df_scaler = pd.DataFrame(0, index=[0], columns=scaler_cols)
        for col in scaler_cols:
            if col in input_data.columns:
                df_scaler[col] = input_data[col].values[0]
        
        scaled_array = scaler.transform(df_scaler)
        df_scaled = pd.DataFrame(scaled_array, columns=scaler_cols)
    else:
        df_scaled = input_data

    # 5. ΕΥΘΥΓΡΑΜΜΙΣΗ ΜΕ ΤΟ ΜΟΝΤΕΛΟ
    model_cols = getattr(model, 'feature_names_in_', None)
    if model_cols is not None:
        # Δίνουμε στο μοντέλο τις scaled στήλες + τα Dummies
        final_X = pd.DataFrame(0, index=[0], columns=model_cols)
        for col in model_cols:
            if col in df_scaled.columns:
                final_X[col] = df_scaled[col].values[0]
            elif col in input_data.columns:
                final_X[col] = input_data[col].values[0]
    else:
        final_X = df_scaled

    # 6. Πρόβλεψη
    prob = model.predict_proba(final_X)[0][1]
    prediction = int(model.predict(final_X)[0])
    label = "Purchased" if prediction == 1 else "Not Purchased" 
    
    return f"Prediction: {label} (Probability of purchase: {prob:.1%})"

@tool
def ecommerce_calculator(price: float, discount_percentage: float, tax_rate: float) -> str:
    """
    Calculates the final price of an e-commerce product after applying a discount and tax.
    Use this when the user asks to calculate final prices, discounts, or total costs.
    """
    discount_amount = price * (discount_percentage / 100)
    price_after_discount = price - discount_amount
    tax_amount = price_after_discount * (tax_rate / 100)
    final_price = price_after_discount + tax_amount
    
    return f"The calculation is complete. The final price after a {discount_percentage}% discount and {tax_rate}% tax is ${final_price:.2f}"