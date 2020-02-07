import pandas as pd
import Load_Data as ld
import Data_Preprocessor as dp
import ML_ModelBuild_Predict_Evaluate as ml
import ML_Predictyield_Newdata as mp

def main():
    df=ld.load_Data()
    df_preprocessed = dp.preprocess_Data(df)
    ml.perform_Machinelearningtasks(df_preprocessed)
    mp.process_predict_newdata()

if __name__ == "__main__":
    main()
