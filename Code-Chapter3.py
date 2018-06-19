import numpy as np
import pandas as pd

#Read Data
datadf = pd.read_csv('data.csv',sep=';', header='infer')

#remove line in data where Age=NaN
datadf = datadf[np.isfinite(datadf['Age'])]
#remove line in data where Python_user=NaN
datadf = datadf.dropna(subset=['Python_user'])
#Replace values
datadf['Gender'] = datadf['Gender'].replace(['male','female'],[0,1])
#Replace values
datadf['Python_user'] = datadf['Python_user'].replace(['no','yes'],[0,1])

#Output example sample before pre-processing
datadf.ix[0:9,['R_user']]
#Replace values
datadf['R_user'] = datadf['R_user'].replace(['no','yes'],[0,1])
#Output example sample after pre-processing
datadf.ix[0:9,['R_user']]

