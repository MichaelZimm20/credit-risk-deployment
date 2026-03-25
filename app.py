'''
Using Gradio to create a simple web interface for the LightGBM model used to predict credit risk defaulters. 

- It deploys the LightGBM model refined from Module 8, and this 
app will allow users to upload raw CSV files  or use preloaded examples from the UCI Credit Risk dataset.\
    

After creating app.py this will be used to deploy the app on Hugging Face Spaces.

References:

- https://huggingface.co/docs/hub/en/spaces-sdks-gradio
'''


# =============== IMPORTS ==============
import gradio as gr # for creating the web interface hosted on Hugging Face Spaces
import pandas as pd # to handle dataframes 
import joblib # for loading the trained LightGBM model, saved from module 8 


# ======== LOAD THE MODEL AND FEATURES ========

# LOAD THE TRAINED LIGHTGBM MODEL
model = joblib.load('lgbm_credit_risk_model.joblib')

# load the features used in the training model
feature_names = joblib.load('lgbm_credit_risk_features.joblib')



# ====== PREDICTION FUNCTION FOR USER SUBMITTED CSV DATA FILES =========
'''
        The goal of this funcvtion is to take in a user uploaded CSV file,
        process and validate it, run the predictions using the model,
        and return it as a dataframe using pandas
        
        - Found that Gradio has its own built in error handling.
            - gr.Error() will raise errors caught by Gradio and display them in the web interface, instead of crashing the app (and or using ValueError Exception)
    '''
def predict_credit_risk(file):
    
    # using the try accept to catch any errors from file uploading and reading into a pandas dataframe
    try:
        # read the user uploaded CSV file into a pandas dataframe
        user_upload_df = pd.read_csv(file)
    except Exception as e:
        raise gr.Error(f'Could not read file. Ensure is is a valid CSV file. Error: {e}')
    
    # Validation Checks 
    if not all(features in user_upload_df.columns for features in feature_names):
        missing_features = set(feature_names) - set(user_upload_df.columns)
        raise gr.Error(f'The uploaded CSV file is missing the following required features: {missing_features}')
    
    # --- Filter dataframe to only include the 23 features from the trained model -----
    df_filtered = user_upload_df[feature_names]
    
    
    ''' Using similair code from train_model function from module 8 '''
    # --- Run the predictions using the LightGBM model ---
    probabilities = model.predict_proba(df_filtered)[:, 1] # get the probability of the defaulter class (1)
    
    #--- Create binary predictions with threshold of 0.5 ---
    predictions = (probabilities >= 0.5).astype(int)               # thresholding at 0.5 for binary classification   
    
    # --- Create a results dataframe to return to the user ---
    
    # make a copy of the users original dataframe, to ensure we do not modify the original data and allow for predictions and probabilities to be added to the new df
    results_df = df_filtered.copy()
    
    # add the predictions and probabilties to the dataframe 
    results_df['Prediction'] = pd.Series(predictions).map({0: 'Non-Defaulter', 1: 'Defaulter'}) # map the binary predictions with more readable format instead of just 0 or 1  
    results_df['Default Probability'] = pd.Series(probabilities).map('{:.4f}'.format) # format to 4 decimal places for readability   
    
    # return the updated dataframe with the predictions and probabilities columns for the user 
    #print(results_df.head()) # print the first few rows of the results dataframe to check the output format
    return results_df
    

# ===================== TEST FUNCTION ======================================
# test the function 
#csv_file = 'files/sample_input.csv' # path to a sample input CSV file with the required features for testing
#predict_credit_risk(csv_file)
# ==========================================================================


# ===================== GRADIO APP INTERFACE ================================
# Create Gradio interface

gradio_app = gr.Interface(
    predict_credit_risk, # the function to run when the user submits a file
    gr.File(label='Upload CSV file with credit risk features', file_types=['.csv']), # input file upload, only allow CSV files
    gr.DataFrame(label='Predictions & Probabilities added to output dataframe'),
    title="Credit Risk Defaulter Prediction App",
    description=f"Upload your CSV file with required features to get back predicitons and probabilities of users at risk of defaulting on their credit. Required Features: (\n{len(feature_names)}): {', '.join(feature_names)}",
    examples=[["files/sample_input.csv"], ["files/sample_small.csv"]]
)
    
# launch the app ( will be deployed on Hugging Face Space)
if __name__ == "__main__":
    gradio_app.launch()      
        
    
