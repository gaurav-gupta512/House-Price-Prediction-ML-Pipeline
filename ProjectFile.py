import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

# building pipeline for incoming data for the model
def buildPipeline(numericalAttributes,categoricalAttributes):
    numericalPipeline = Pipeline([
        ('imputation',SimpleImputer(strategy='median')),
        ('scaling',StandardScaler())
    ])

    categoricalPipeline = Pipeline([
        ('oneHotEncoding',OneHotEncoder(handle_unknown='ignore')) #'ignore' ignores unseen data types
    ])

    fullPipeline = ColumnTransformer([
        ('numericalPipeline',numericalPipeline,numericalAttributes),
        ('categoricalPipeline',categoricalPipeline,categoricalAttributes)
    ])

    return fullPipeline

# checking if model exists, if yes. infer data in the model and get results
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    inputFile = pd.read_csv('input.csv') # inputting a csv file with only test data 
    transformedInput = pipeline.transform(inputFile)
    predictions = model.predict(transformedInput)
    inputFile['Predictions'] = predictions

    inputFile.to_csv('output.csv',index=False) # saving all predictions in a new csv file
    print('inference complete.predictions done')

else:
    houseData = pd.read_csv('housing.csv') # complete data that will be split into training and testing sets

    houseData['income_category'] = pd.cut(houseData['median_income'],
                                      bins=[0,2,4,6,8,np.inf],
                                      labels=[1,2,3,4,5])
    
    splitter = StratifiedShuffleSplit(n_splits=1,
                                        test_size=0.2,
                                        random_state=42)

    for trainingDataIndex,_ in splitter.split(houseData,houseData['income_category']):
        houseData = houseData.loc[trainingDataIndex].drop('income_category',axis=1)

    houseDataLabels = houseData['median_house_value'].copy() # seperating labels
    houseDataFeatures = houseData.drop('median_house_value',axis=1) # seperating predictor/keeping only features

    numericalAttributes = houseDataFeatures.drop('ocean_proximity',axis=1).columns.tolist()
    categoricalAttributes = ['ocean_proximity']

    pipeline = buildPipeline(numericalAttributes,categoricalAttributes)
    houseDataTransformed = pipeline.fit_transform(houseDataFeatures)

    model = RandomForestRegressor(random_state=42).fit(houseDataTransformed,houseDataLabels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print('model trained and saved')
