# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:22:39 2016

@author: Rui Sarmento
"""

#read csv with survey data
import pandas as pd
data = pd.read_csv('newdata.csv', sep=';')
survey_data = data.ix[:,7:17]

#summary of the variables
survey_data.describe()

#Correlation matrix
survey_data_corr = survey_data.corr(method='spearman')
survey_data_corr

#Bartlett Sphericity Test
#Generate Identity Matrix
import numpy as np
#10x10 Identity Matrix
indentity = np.identity(10)

#The Bartlett test
import math as math
import scipy.stats as stats
n = survey_data.shape[0]
p = survey_data.shape[1]
chi2 = -(n-1-(2*p+5)/6)*math.log(np.linalg.det(survey_data_corr))
ddl = p*(p-1)/2
pvalue = stats.chi2.pdf(chi2 , ddl)
chi2
ddl
pvalue



#KMO Test

### KMO Measure
import numpy as np
import math as math

dataset_corr = survey_data_corr
### a) Global KMO
# Inverse of the correlation matrix
# dataset_corr is the correlation matrix of the survey results
corr_inv = np.linalg.inv(dataset_corr)
# number of rows and number of columns
nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
# Partial correlation matrix
A = np.ones((nrow_inv_corr,ncol_inv_corr))
for i in range(0,nrow_inv_corr,1):
    for j in range(i,ncol_inv_corr,1):
        #above the diagonal
        A[i,j] = - (corr_inv[i,j])/(math.sqrt(corr_inv[i,i] * corr_inv[j,j]))
        #below the diagonal
        A[j,i] = A[i,j]
#transform to an array of arrays ("matrix" with Python)
dataset_corr = np.asarray(dataset_corr)
#KMO value
kmo_num = np.sum(np.square(dataset_corr))-np.sum(np.square(np.diagonal(dataset_corr)))
kmo_denom = kmo_num + np.sum(np.square(A))-np.sum(np.square(np.diagonal(A)))
kmo_value = kmo_num / kmo_denom
print(kmo_value)
### b) KMO per variable
#creation of an empty vector to store the results per variable. The size of the vector is equal to the number
#...of variables
kmo_j = [None]*dataset_corr.shape[1]
for j in range(0, dataset_corr.shape[1]):
    kmo_j_num = np.sum(dataset_corr[:,[j]]**2) - dataset_corr[j,j]**2
    kmo_j_denom = kmo_j_num + np.sum(A[:,[j]]**2) - A[j,j]**2
    kmo_j[j] = kmo_j_num / kmo_j_denom
print(kmo_j)

#Kaiser Criterion
np.linalg.eig(survey_data_corr)

#Scree Plot (calling R functions from Python)
#See https://sites.google.com/site/aslugsguidetopython/data-analysis/pandas/calling-r-from-python
#See http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2 to install rpy2
import rpy2 as rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

#Changing R's directory to where is the data
ro.r('setwd("C:/Users/Rui Sarmento/Documents/Livro Cybertech/Dados e Code")')

#Reading the data with R
ro.r('data_df <- read.csv("data.csv",sep=";")')

#retrieving the correlation matrix of the survey answers
ro.r('correlation <- cor(data_df[,paste("Q",1:10,sep="")], method="spearman")')

### Scree plot criterion
psych = importr('psych')
#scree function call with R
ro.r('scree(correlation, hline=-1)') # hline=-1 draw a horizontal line at -1')



#Explained Variance Criteria
#See http://www.dummies.com/how-to/content/data-science-using-python-to-perform-factor-and-pr.html
from sklearn.decomposition import PCA
import pandas as pd
pca = PCA().fit(survey_data)
pca.explained_variance_ratio_

#PCA (using R)
### Principal Component method
import rpy2 as rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

#Changing R's directory to where is the data
ro.r('setwd("C:/Users/Rui Sarmento/Documents/Livro Cybertech/Dados e Code")')

#Reading the data with R
ro.r('data_df <- read.csv("data.csv",sep=";")')

#retrieving the correlation matrix of the survey answers
ro.r('correlation <- cor(data_df[,paste("Q",1:10,sep="")], method="spearman")')

#It uses psych R's package
ro.r('library (psych)')

#calling function principal with R
print(ro.r('principal(correlation,nfactors=3, rotate="none")'))


#PCA - new correlation (with R)
#new correlation matrix
ro.r('new.correlation<-correlation[!(colnames(correlation) %in% c("Q6", "Q8", "Q10")),!(rownames(correlation) %in% c("Q6", "Q8", "Q10"))]')

ro.r('library(psych)')

#calling function principal with R
print(ro.r('principal(new.correlation,nfactors=3, rotate="none")'))


#PCA - Varimax rotation (with R)
ro.r('library(psych)')
#calling function principal with varimax rotation (with R)
print(ro.r('principal(correlation,nfactors=3, rotate="varimax")'))

#Cronbach Alphas function
def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))
    
#PC1 (Q1, Q6, Q7, Q9, Q10)
CronbachAlpha(np.matrix(survey_data[[0,5,6,8,9]].transpose()))

#PC2 (Q4,Q5,Q8)
CronbachAlpha(np.matrix(survey_data[[3,4,7]].transpose()))

#PC3 (Q2, Q3)
CronbachAlpha(np.matrix(survey_data[[1,2]].transpose()))






