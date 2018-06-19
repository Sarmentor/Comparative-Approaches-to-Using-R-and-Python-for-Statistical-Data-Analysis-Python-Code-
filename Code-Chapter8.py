# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:22:39 2016

@author: Rui Sarmento
"""

#read csv with survey data
import pandas as pd
data = pd.read_csv('newdata.csv', sep=';')

#import libraries
import scipy.spatial.distance as sp
#filtering some data from 2015 
new_data = data[data['Year'] >= 2015]
#filtering survey data
survey_data = new_data.ix[:,7:17]
#calculate distance
X = sp.pdist(survey_data, 'euclidean')
X

#Hierarchical clustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#single linkage
Z_single = linkage(X, 'single')
#complete linkage
Z_complete = linkage(X, 'complete')
#Dendograms

# calculate full dendrogram (single)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z_single,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# calculate full dendrogram (complete)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

#clustering with kmeans in Python
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import whiten
#Normalize variables values
std_survey_data = whiten(survey_data, check_finite=True)
#kmeans algorithm
kmeans2(std_survey_data,2, iter=10)