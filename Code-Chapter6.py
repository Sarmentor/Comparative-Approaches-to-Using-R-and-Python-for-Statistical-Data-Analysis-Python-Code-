# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:22:39 2016

@author: Rui Sarmento
"""

#read csv with survey data
import pandas as pd
datadf = pd.read_csv('newdata.csv', sep=';')

#example
datadf.ix[0:9,['R_user']]


import numpy as np

#remove line in data where Age=NaN
datadf = datadf[np.isfinite(datadf['Age'])]
#remove line in data where Python_user=NaN
datadf = datadf.dropna(subset=['Python_user'])

#Replace values 
datadf['Gender'] = datadf['Gender'].replace(['male','female'],[0,1])

#Replace values
datadf['Python_user'] = datadf['Python_user'].replace(['no','yes'],[0,1])

#Replace values 
datadf['R_user'] = datadf['R_user'].replace(['no','yes'],[0,1])

### Influence of variables Gender, Python_user, R_user, Age in the number of publications
#Import libraries/modules
import numpy as np
import statsmodels.formula.api as smf

# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('Publications ~ Gender + Python_user + R_user + Age', data=datadf).fit()

# Inspect the results
print(results.summary())



#EXTRA - NOT IN THE BOOK (Predicting with Python Regression Models)
#Separate data in train and test sets

#train = datadf.sample(frac=0.8, random_state=1)
#test = datadf.loc[~datadf.index.isin(train.index)]

#declare the model and fit
#lr.fit(train[["Publications"]], train[["Gender","Python_user","R_user","Age"]])

#get regression coeficients, residues and intercept
#lr.coef_
#lr.intercept_
#lr.residues_

#predict number of publications of test set
#predictions = lr.predict(test[["Publications"]])

#R^2 Score of prediction
#lr.score(test[["Publications"]],predictions)


