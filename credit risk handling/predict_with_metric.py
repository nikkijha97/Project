import time
import numpy as np
import logging
import pickle
import pandas as pd
import numpy as np
import cml.models_v1 as models
import cml.metrics_v1 as metrics

# Load the saved pipeline
model = pickle.load(open('fitted_pipeline.pkl', 'rb'))

# Define the predict function
@models.cml_model(metrics = True)
def predict(args):
    """
    Predict function for deployment. Accepts input as a dictionary of arguments
    and returns the predicted probability of default.
    """
    #track input
    metrics.track_metric("input", args)
    # Convert incoming JSON args into a DataFrame
    df = pd.DataFrame([args])
    
    # Verify that required columns exist
    required_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {"error": f"Missing required columns: {missing_columns}"}
    
    # Add derived columns (to ensure proper structure for the pipeline)
    for variable in required_columns:
        if variable in df.columns:
            df[variable + "_no_card_use"] = (df[variable] == -2).astype(int)
            df[variable + "_payed_off"] = (df[variable] == -1).astype(int)
            df[variable] = df[variable].clip(lower=0)  # Replace negative values with 0

    # Ensure column order matches the training data (pipeline handles scaling/encoding)
    try:
        result = model.predict_proba(df)  # Predict probabilities
        probability_of_default = round(result[0][1].item(), 2)  # Get the probability for class 1 (default)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

    # Return the result as a JSON response
    metrics.track_metric("predictions", probability_of_default)
    return {"probability_of_default": probability_of_default}
