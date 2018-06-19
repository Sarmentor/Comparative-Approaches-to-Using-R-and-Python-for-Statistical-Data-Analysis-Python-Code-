from pandas import *
import pandas as pd

datadf =  pd.read_csv('data.csv', sep=';')

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


import statistics as stats
#Numerical variables median
stats.median(datadf['Publications'])

#first we have to take NaN values and substitute them for the mean of the variable
datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
stats.median(datadf['Age'])

#categorical variables mode
stats.mode(datadf['Python_user'])
stats.mode(datadf['R_user'])
stats.mode(datadf['Tasks'])
stats.mode(datadf['Gender'])

#Change columns names
#datadf.columns = ['id','Gender','Python_user','R_user','Age','Publications','Tasks']

#Replace values 
datadf['Gender'] = datadf['Gender'].replace([1,2],['male','female'])

#Replace values
datadf['Python_user'] = datadf['Python_user'].replace(['0','1',' '],['no','yes','NaN'])

#Replace values 
datadf['R_user'] = datadf['R_user'].replace([0,1],['no','yes'])

#Replace values 
datadf['Tasks'] = datadf['Tasks'].replace([1,2,3],['Phd_Supervisor','Postdoctoral_research','PhD_Student'])


#Frequencies of Gender, Python_user, R_user and Tasks
print(datadf['Gender'].value_counts())

#cumulative and relative frequencies
gender_datadf = datadf['Gender']
#group by gender
gender_datadf = pd.DataFrame(gender_datadf.value_counts(sort=True))
#create new column with cumulative sum
gender_datadf['cum_sum'] = gender_datadf['Gender'].cumsum()
#create new column with relative frequency
gender_datadf['cum_perc'] = 100*gender_datadf['cum_sum']/gender_datadf['Gender'].sum()
gender_datadf

print(datadf['Python_user'].value_counts())

#cumulative and relative frequencies
python_datadf = datadf['Python_user']
#group by Python users
python_datadf = pd.DataFrame(python_datadf.value_counts(sort=True, dropna =False))
#create new column with cumulative sum
python_datadf['cum_sum'] = python_datadf['Python_user'].cumsum()
#create new column with relative frequency
python_datadf['cum_perc'] = 100*python_datadf['cum_sum']/python_datadf['Python_user'].sum()
python_datadf


print(datadf['R_user'].value_counts())

#cumulative and relative frequencies
r_datadf = datadf['R_user']
#group by R users
r_datadf = pd.DataFrame(r_datadf.value_counts(sort=True))
#create new column with cumulative sum
r_datadf['cum_sum'] = r_datadf['R_user'].cumsum()
#create new column with relative frequency
r_datadf['cum_perc'] = 100*r_datadf['cum_sum']/r_datadf['R_user'].sum()
r_datadf

print(datadf['Tasks'].value_counts())

#cumulative and relative frequencies
tasks_datadf = datadf['Tasks']
#group by tasks
tasks_datadf = pd.DataFrame(tasks_datadf.value_counts(sort=True))
#create new column with cumulative sum
tasks_datadf['cum_sum'] = tasks_datadf['Tasks'].cumsum()
#create new column with relative frequency
tasks_datadf['cum_perc'] = 100*tasks_datadf['cum_sum']/tasks_datadf['Tasks'].sum()
tasks_datadf


#Frequency of a numeric variable
print(datadf['Age'].value_counts())
print(datadf['Publications'].value_counts())

#cumulative and relative frequencies
pubs_datadf = datadf['Publications']
#group by publications
pubs_datadf = pd.DataFrame(pubs_datadf.value_counts(sort=True))
#create new column with cumulative sum
pubs_datadf['cum_sum'] = pubs_datadf['Publications'].cumsum()
#create new column with relative frequency
pubs_datadf['cum_perc'] = 100*pubs_datadf['cum_sum']/pubs_datadf['Publications'].sum()
pubs_datadf


#Class Division
buckets = [0,10, 20, 30, 40,70]
table = np.histogram(datadf['Publications'], bins=buckets)
print(table)
#equal size
table = np.histogram(datadf['Publications'], bins=11, range=(0, 70))
print(table)

#Pie Chart Gender
import matplotlib.pyplot as plt
import pandas as pd
#Pie chart labels
label_list = datadf['Gender'].value_counts(sort=False).index
plt.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")

#To show the percentage of each pie slice, pass an output format to the autopctparameter 
plt.pie(datadf['Gender'].value_counts(sort=False),labels=label_list,autopct="%1.1f%%") 
plt.title("Researchers Gender")
plt.show()

#Pie Chart Python_user
import matplotlib.pyplot as plt
import pandas as pd
#Pie chart labels
label_list = datadf['Python_user'].value_counts(sort=False).index
plt.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")

#To show the percentage of each pie slice, pass an output format to the autopctparameter 
plt.pie(datadf['Python_user'].value_counts(sort=False),labels=label_list,autopct="%1.1f%%") 
plt.title("Researchers Python Users")
plt.show()

#Pie Chart R_user
import matplotlib.pyplot as plt
import pandas as pd
#Pie chart labels
label_list = datadf['R_user'].value_counts(sort=False).index
plt.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")

#To show the percentage of each pie slice, pass an output format to the autopctparameter 
plt.pie(datadf['R_user'].value_counts(sort=False),labels=label_list,autopct="%1.1f%%") 
plt.title("Researchers R Users")
plt.show()


#Boxplot Publications
import matplotlib.pyplot as plt
import pandas as pd
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
#Variable
ax.boxplot(datadf['Publications'],showfliers=True, flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                  linestyle='none'))
plt.title('Publications Boxplot')
plt.show()

#Boxplot Age
#first we have to take NaN values and substitute them for the mean of the variable
datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
import matplotlib.pyplot as plt
import pandas as pd
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
#Variable
ax.boxplot(datadf['Age'],showfliers=True, flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                  linestyle='none'))
plt.title('Age Boxplot')
plt.show()

#Boxplot Publications by Age
#now with pandas package
import pandas as pd
#first we have to take NaN values and substitute them for the mean of the variable
datadf['Age']=datadf['Age'].replace('nan',datadf['Age'].mean())
#get only Publications and Age variables
new_datadf = datadf.ix[:,['Age','Publications']]
#boxplot 
new_datadf.boxplot(by='Age',rot = 90)

#Publications Histogram
fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(datadf['Publications'], facecolor='green') # Here you can play with number of bins
#limits of x axis
ax.set_xlim(0, 70)
#limits of y axis
ax.set_ylim(0,80)
#Set grid on
ax.grid(True)
#Labels and Title
plt.title('Publications distribution')
plt.xlabel('Publications')
plt.ylabel('#Reseachers')
#Finally, show plot
plt.show()

#tasks Barplot
var = datadf['Tasks'].value_counts(sort=False); #grouped sum of Tasks
plt.figure();
plt.ylabel('Number of Reseachers')
plt.title("Counting Reseacher's Tasks")
var.plot.bar();
plt.show();