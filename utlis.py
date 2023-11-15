import numpy as np
import pandas as pd
# spliting data
from sklearn.model_selection import train_test_split
# preprocessing and feature transformation 
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import PowerTransformer,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn_features.transformers import DataFrameSelector
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"],axis=1,inplace=True)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df.columns=df.columns.str.lower()

df.drop_duplicates(inplace=True)
X=df.drop("churn",axis=1)
y=df["churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,shuffle=True,random_state=45)
important_columns=['contract','tenure','monthlycharges','totalcharges','techsupport','onlinesecurity','paperlessbilling','paymentmethod','internetservice','dependents']
X_train=X_train.loc[:,important_columns]
num_cols=X_train.select_dtypes(include="number").columns.to_list()
cat_cols=X_train.select_dtypes(exclude="number").columns.to_list()



cat_pipe=Pipeline(steps=[
    ("selector",DataFrameSelector(cat_cols)),
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder())
])
num_pipe=Pipeline(steps=[
    ("selector",DataFrameSelector(num_cols)),
    ("impute",SimpleImputer(strategy="mean")),
    ("tranform",PowerTransformer(standardize=True))
])
all_pipe=FeatureUnion(transformer_list=[
    ("caterogical_pipline",cat_pipe),
    ("numerical_pipline",num_pipe)
])


_ =all_pipe.fit(X_train)


def process_new(X_new):
    ''' This Function is to apply the pipeline to user data. Taking a list.
    
    Args:
    *****
        (X_new: List) --> The users input as a list.

    Returns:
    *******
        (X_processed: 2D numpy array) --> The processed numpy array of userf input.
    '''
    
    ## To DataFrame
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    ## Adjust the Datatypes
    df_new['contract'] = df_new['contract'].astype('str')
    df_new['tenure'] = df_new['tenure'].astype('int')
    df_new['monthlycharges'] = df_new['monthlycharges'].astype('float')
    df_new['totalcharges'] = df_new['totalcharges'].astype('float')
    df_new['techsupport'] = df_new['techsupport'].astype('str')
    df_new['onlinesecurity'] = df_new['onlinesecurity'].astype('str')
    df_new['paperlessbilling'] = df_new['paperlessbilling'].astype('str')
    df_new['paymentmethod'] = df_new['paymentmethod'].astype('str')
    df_new['internetservice'] = df_new['internetservice'].astype('str')
    df_new['dependents'] = df_new['dependents'].astype('str')



    ## Apply the pipeline
    X_processed = all_pipe.transform(df_new)


    return X_processed






