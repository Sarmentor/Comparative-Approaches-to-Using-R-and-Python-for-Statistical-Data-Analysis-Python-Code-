#from statsmodel import *
import pandas as pd

datadf =  pd.read_csv('newdata.csv', sep=';')

print(list(datadf.columns.values))
print(list(datadf.index))


#Age
print("\nAge Variable: \n")
print("Number of elements: {0:8.0f}".format(len(datadf['Age'])))
print("Minimum: {0:8.3f} Maximum: {1:8.3f}".format(datadf['Age'].min(), datadf['Age'].max()))
print("Mean: {0:8.3f}".format(datadf['Age'].mean()))
print("Variance: {0:8.3f}".format(datadf['Age'].var()))
print("Standard Deviation : {0:8.3f}".format(datadf['Age'].std()))

#Publications
print("\nPublications: \n")
print("Number of elements: {0:8.0f}".format(len(datadf['Publications'])))
print("Minimum: {0:8.3f} Maximum: {1:8.3f}".format(datadf['Publications'].min(), datadf['Publications'].max()))
print("Mean: {0:8.3f}".format(datadf['Publications'].mean()))
print("Variance: {0:8.3f}".format(datadf['Publications'].var()))
print("Standard Deviation : {0:8.3f}".format(datadf['Publications'].std()))

#Change columns names
datadf.columns = ['id', 'Gender', 'Python_user', 'R_user', 'Age', 'Publications', 'Tasks', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Year'
]

#Replace values 
datadf['Gender'] = datadf['Gender'].replace([1,2],['male','female'])

#Replace values
datadf['Python_user'] = datadf['Python_user'].replace(['0','1',' '],['no','yes','NaN'])

#Replace values 
datadf['R_user'] = datadf['R_user'].replace([0,1],['no','yes'])

#Replace values 
datadf['Tasks'] = datadf['Tasks'].replace([1,2,3],['Phd_Supervisor','Postdoctoral_research','PhD_Student'])

#Normality Tests
import scipy
from scipy import stats
import numpy
from numpy import random as dist

datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
print(stats.shapiro(datadf['Age']))
print(stats.shapiro(datadf['Publications']))


normed_data_age=(datadf['Age']-datadf['Age'].mean())/datadf['Age'].std()
normed_data_pubs=(datadf['Publications']-datadf['Publications'].mean())/datadf['Publications'].std()

print(stats.kstest(normed_data_age, cdf='norm'))#cdf = stats.norm.cdf))
print(stats.kstest(normed_data_pubs, cdf='norm'))#cdf = stats.norm.cdf))

#KSmirnof gives strange results with Python
#...thus, we are doing it with R
from pandas import *
import rpy2 as rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

#Changing R's directory to where is the data
ro.r('setwd("C:/Users/Rui Sarmento/Documents/Livro Cybertech/Dados e Code")')

#Reading the data with R
ro.r('data_df <- read.csv("newdata.csv",sep=";")')

# Reading the R's package zoo, needed to apply na.aggregate
ro.r('library(zoo)')

#Kolmogorov-smirnov Normal Distribution test
print(ro.r('ks.test(na.aggregate(data_df$Age), rnorm(200, mean(na.aggregate(data_df$Age)), sd(na.aggregate(data_df$Age))))'))

#Kolmogorov-Smirnov Normal Distribution test
print(ro.r('ks.test(data_df$Publications, rnorm(200, mean(data_df$Publications)))'))


#QQ PLOT with python
import pylab 
import scipy.stats as stats

#Age  
stats.probplot(datadf['Age'], dist="norm", plot=pylab)
pylab.show()

#Publications
stats.probplot(datadf['Publications'], dist="norm", plot=pylab)
pylab.show()

#TTEST Age (male vs female)
datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
print(stats.ttest_ind(datadf['Age'][datadf['Gender'] == "male"],datadf['Age'][datadf['Gender'] == "female"]))

#ANOVA
#Levene Test Age ~ Tasks
#note: test with na Age substituted by mean(datadf['Age']) 
print(stats.levene(datadf['Age'][datadf['Tasks']=="PhD_Student"],datadf['Age'][datadf['Tasks']=="Phd_Supervisor"],datadf['Age'][datadf['Tasks']=="Postdoctoral_research"], center = 'median'))

#One-Way Anova
print(stats.f_oneway(datadf['Age'][datadf['Tasks']=="PhD_Student"],datadf['Age'][datadf['Tasks']=="Phd_Supervisor"],datadf['Age'][datadf['Tasks']=="Postdoctoral_research"]))

#One-Way Anova with R for Welch correction
from pandas import *
import rpy2 as rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

#Changing R's directory to where the data is 
ro.r('setwd("C:/Users/Rui Sarmento/Documents/Livro Cybertech/Dados e Code")')

#Reading the data with R
ro.r('data_df <- read.csv("data.csv",sep=";")')

#get NA values equal to Age's mean with function na.aggregate with zoo package
ro.r('library(zoo)')
ro.r('library(stats)')
#ANOVA with Welch correction (var.equal = FALSE)
print(ro.r('oneway.test(na.aggregate(data_df$Age)~Tasks, data = data_df, na.action=na.omit, var.equal=FALSE)'))

#Games-Howell post-hoc test (with R)
#See https://sites.google.com/site/aslugsguidetopython/data-analysis/pandas/calling-r-from-python
from pandas import *
import rpy2 as rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

### Games-Howell test for multiple comparisons

#Changing R's directory to where the data is 
ro.r('setwd("C:/Users/Rui Sarmento/Documents/Livro Cybertech/Dados e Code")')

#Reading the data with R
ro.r('data_df <- read.csv("data.csv",sep=";")')

# Reading the R's package zoo, needed to apply na.aggregate
ro.r('library(zoo)')

#store the data in an auxiliary variable for editing
ro.r('data_df_new <- data_df')

# Substitution of the missing values by the mean of the variable
ro.r('data_df_new$Age <- na.aggregate(as.numeric(data_df$Age), by="Age", FUN = mean)')

# Reading the R's package userfriendlyscience, needed to apply Games-Howell test
#this time the loading is done with the importr function
science = importr('userfriendlyscience')

#Games-Howell test 
print(ro.r('oneway(y=data_df_new$Age, x = data_df$Tasks, posthoc="games-howell", means=T, fullDescribe=T, levene=T,plot=T, digits=2, pvalueDigits=3, conf.level=0.95)'))



#Tukey
#note: change of package
from statsmodels.stats.multicomp import pairwise_tukeyhsd
datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
tukey = pairwise_tukeyhsd(endog=datadf['Age'], groups=datadf['Tasks'], alpha=0.05)
print(tukey.summary())

#Mann-Whitney
print(stats.mannwhitneyu(datadf['Publications'][datadf['Gender']=="male"],datadf['Publications'][datadf['Gender']=="female"]))

#Kruskal-Wallis
from scipy.stats.mstats import kruskalwallis
print(kruskalwallis(datadf['Publications'][datadf['Tasks']=="PhD_Student"],datadf['Publications'][datadf['Tasks']=="Phd_Supervisor"],datadf['Publications'][datadf['Tasks']=="Postdoctoral_research"]))

#Mann-Whitney
print(stats.mannwhitneyu(datadf['Publications'][datadf['Tasks']=="PhD_Student"],datadf['Publications'][datadf['Tasks']=="Postdoctoral_research"]))
print(stats.mannwhitneyu(datadf['Publications'][datadf['Tasks']=="PhD_Student"],datadf['Publications'][datadf['Tasks']=="Phd_Supervisor"]))
print(stats.mannwhitneyu(datadf['Publications'][datadf['Tasks']=="Phd_Supervisor"],datadf['Publications'][datadf['Tasks']=="Postdoctoral_research"]))

#Pearson's Chi-Squared
import scipy.stats as stats
import numpy as np
import pandas as pd
#R users by Gender
table_r = datadf.pivot_table(index='Gender',columns='R_user', values = 'id',aggfunc='count')
print(stats.chi2_contingency(table_r))

#Python users by Gender
table_py = datadf.pivot_table(index='Gender',columns='Python_user', values = 'id',aggfunc='count')
print(stats.chi2_contingency(table_py))

#Tasks by Gender
table_tasks = datadf.pivot_table(index='Gender',columns='Tasks', values = 'id',aggfunc='count')
print(stats.chi2_contingency(table_tasks))

#Spearman Correlation Test
print(stats.spearmanr(datadf['Publications'],datadf['Age']))