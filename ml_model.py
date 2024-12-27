import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("loan_approval_dataset.csv")

df.head()

df.describe()

df.info()

df.isnull().sum()

import warnings
warnings.filterwarnings('ignore')

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

threshold = 1

# Identify outliers
outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))

# Count the number of outliers in each column
outliers_count = outliers.sum()
outliers_count

df = df[~outliers.any(axis=1)]

df.drop(['loan_id'],axis=1)

df.nunique()

df.select_dtypes(include='object').nunique()

df.columns

df[' education'].value_counts()

df[' loan_status'].value_counts()

df[' self_employed'].value_counts()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df[' education'] = label_encoder.fit_transform(df[' education'])
df[' self_employed'] = label_encoder.fit_transform(df[' self_employed'])
df[' loan_status'] = label_encoder.fit_transform(df[' loan_status'])

df.head()

education_counts = df[' education'].value_counts()

x = df.drop(columns = [' loan_status'])
y = df[' loan_status']

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_scaled = sc.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=9)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initiate the Model
rfc = RandomForestClassifier()

# Fit the Model
rfc.fit(x_train, y_train)

# Make the pickle file of our model
pickle.dump(rfc, open("model.pkl" , "wb"))






