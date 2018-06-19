# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

3+2*5
x = 3**2
# OR
#import math module
import math as math
#call pow function
x=math.pow(3,2)
x
#call to sqrt function
math.sqrt(x)

import array as array
my_array = array.array('i',(range(1,11)))
my_array

[x+2 for x in my_array]

#new array
my_array2 = array.array('i',(range(3,13)))

import numpy as np
new_array = np.add(my_array, my_array2)
new_array

import numpy as np
new_array2 = np.add(my_array, 2)
new_array2

char_array = ['String1','String2','String3']
char_array

#first and second position
char_array[0:2]
#first and third position of the array
char_array[0::2]

#length of array
len(my_array)

#fourth position of array
my_array[3]

#first position of the array
my_array[0]

def add(x,y):
   return x+y
   
add(x=2,y=2)

import pandas as pd
my_dataframe = pd.read_csv('test.csv')
my_dataframe

my_dataframe(2,2)

students = ["John","Mike","Vera","Sophie","Anna","Vera","Vera","Mike","Anna"]
courses = ["Math","Math","Math","Research","Research 2","Research","Research 2","Computation","Computation"]
grades = [13,13,14,16,16,13,17,10,14]

my_grades_dataframe = pd.concat([pd.DataFrame(students,columns=['student']),pd.DataFrame(courses,columns=['course']),pd.DataFrame(grades,columns=['grade'])], axis=1)

my_grades_dataframe['student']

my_grades_dataframe[[0]]

my_grades_dataframe.ix[2,2]

#select the grades > 14
my_grades_dataframe[my_grades_dataframe['grade']>14]

my_grades_dataframe.ix[2,2] = 16
my_grades_dataframe

my_grades_dataframe.info()

my_grades_dataframe.describe()

import numpy as np
my_matrix = np.matrix('0 0 0 0; 0 0 0 0')
my_matrix

my_matrix[1,3] = 5

my_matrix[0,]

my_matrix[0,3]

my_matrix[:,3]

import pandas as pd
my_dataframe.to_excel('my_excel_file_python.xlsx', sheet_name='Sheet1')