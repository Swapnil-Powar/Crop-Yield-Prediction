import pandas as pd

def Display_Data(df):
    print("*****Top 5 rows in a data*****")
    print("\n")
    print(df.head())
    print("\n")
    print("*****Bottom 5 rows in a data*****")
    print("\n")
    print(df.tail())
    print("\n")

def Describe_Data(df):
    print("*****Statistical description of the data*****\n")
    print(df.describe())
    print("\n")
    print("*****Data set Information*****\n")
    print(df.info())


def load_Data():
    df = pd.read_csv('crop_yield_datasource.csv')
    Display_Data(df)
    Describe_Data(df)
    return df

