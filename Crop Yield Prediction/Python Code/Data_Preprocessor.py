import pandas as pd
df = pd.read_csv('crop_yield_datasource.csv')
preprocessedfile = 'crop_yield_datasource_preprocessed.csv'
#check for missing values and handle them
def Handle_MissingData():
    print("*****Number of Missing Data present in dataset*****\n")
    print(df.isnull().sum())
    print("\n")
    print("*****Handling Missing Data present in dataset*****\n")
    print("*****Performing Mean imputation*****\n")
    df_handled_missingData = df.fillna(value=df.mean())
    print(df_handled_missingData.head())
    return df_handled_missingData
#perform data transformation by doing Normalization
def Normalize_Data(df_transformed_data):
    feature_cols = list(df_transformed_data)
    predict_cols = feature_cols[-1:]
    df_target = df_transformed_data[predict_cols]
    df_transformed_data = df_transformed_data.iloc[:,1:9]
    print(df_target.head())
    df_transformed_data = (df_transformed_data - df_transformed_data.mean()) / (df_transformed_data.max() - df_transformed_data.min())
    print("\n*****Normalized Data*****\n")
    print(df_transformed_data.head())
    dataframe_norm = df_transformed_data.join(df_target)
    print(dataframe_norm.head())
    dataframe_norm.to_csv(preprocessedfile, index=False)
    #return fileresponse
    return dataframe_norm
#convert categorical values to continous
def handle_categorical_data(df_handled_categoricalData):
    df_handled_categoricalData["Location"] = df_handled_categoricalData["Location"].replace(["Mysore", "Mandya", "Raichur", "Koppal"], [1,2,3,4])
    print("\n*****Handling Categorical Data*****\n")
    print(df_handled_categoricalData.head())
    return df_handled_categoricalData
def preprocess_Data(df):
    df_handled_missingData=Handle_MissingData()
    df_handled_categoricalData=handle_categorical_data(df_handled_missingData)
    df_transformed_data= Normalize_Data(df_handled_categoricalData)
    return  df_transformed_data