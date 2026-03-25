'''
Running basic tests to ensure the functionaility of the code from app.py 

This will be used during the CI pipeline - Github Actions to verfiy the models functionality before deployment to production.

this will run locally with pytest 
'''

import pytest
import joblib 
import numpy as np
import pandas as pd



# testing model loading 
def test_model_loading():
    try:
        model = joblib.load('lgbm_credit_risk_model.joblib')
        assert model is not None
        assert isinstance(model, object) # checking if the model is an object to ensure it loaded 
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")
        
        
# test feature names load 
def test_feature_names_loading():
    try:
        feature_names = joblib.load('lgbm_credit_risk_features.joblib')
        assert feature_names is not None
        assert isinstance(feature_names, list) # checking if features_names is a list 
        assert len(feature_names) == 23 # checking if number of features is 23 ( the default number of features from the trained model)
    except Exception as e:
        pytest.fail(f"Feature names loading failed: {e}")
        
        
# test the predict_credit_risk function with sample csv datasets 
def test_predict_credit_risk():
    try:
        # load the model and feature names for testing
        model = joblib.load('lgbm_credit_risk_model.joblib')
        feature_names = joblib.load('lgbm_credit_risk_features.joblib')
         
        
        # read the csv using panadas
        sample_csv_df = pd.read_csv('files/sample_input.csv')

        # filter to just feature_names columns 
        feature_df_columns = sample_csv_df[feature_names].columns.tolist()
        assert set(feature_df_columns) == set (feature_names)
        
        # run model.predict_proba on the feature columns 
        probabilities = model.predict_proba(sample_csv_df[feature_names])[:, 1] # get the probability of the defaulter class (1)
        
        # check if the # of probabilities matches the the # of rows in the sample csv 
        assert len(probabilities) == len(sample_csv_df)
        
        # check if probabilities are between 0 and 1 
        assert np.all(probabilities >=0) and np.all(probabilities <= 1)
            
        
    except Exception as e:
        pytest.fail(f"Credit risk prediction failed: {e}")