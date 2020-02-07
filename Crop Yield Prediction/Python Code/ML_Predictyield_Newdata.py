import pandas as pd
import ML_ModelBuild_Predict_Evaluate as mlb

def process_predict_newdata():

    new_data=pd.read_csv('New_crop_data.csv')
    print("*****New data for which crop yield prediction needed*****\n")
    print(new_data.head())
    
    new_data_processed=new_data.iloc[:,1:9]
    new_data_processed["Location"] = new_data_processed["Location"].replace(["Mysore", "Mandya", "Raichur", "Koppal"], [1,2,3,4])
    
    new_data_processed = (new_data_processed - new_data_processed.mean()) / (new_data_processed.max() - new_data_processed.min())
    print("*****Data cleanising result for new data*****\n")
    print(new_data_processed.head())
    
    print("crop yield prediction using Decision Tree model - new data\n")
    new_data['yield prediction from DT']=mlb.DT_regressor.predict(new_data_processed)
    print(new_data.head(15))
    
    print("crop yield prediction using Multiple Linear Regression Model - new data \n")
    new_data['yield prediction from MLR']=mlb.linear_regressor.predict(new_data_processed)
    print(new_data.head(15))
    
    # save result in file
    filename = 'Crop_yield_predictionfornewdata.csv'
    new_data.to_csv(filename, index=False, encoding='utf-8')


