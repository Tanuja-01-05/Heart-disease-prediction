# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from feature_engine.outliers import Winsorizer

heart = pd.read_csv(r"heart.csv")    #reading data

heart.columns  #to display columns in the data 

                         #1st Moment

#to display mean age of the fields in data
heart.age.mean() # Mean of "age" field
heart.sex.mean() # Mean of "sex" field
heart.cp.mean() # Mean of "cp" field
heart.trestbps.mean() # Mean of "trestbps" field
heart.chol.mean() # Mean of "chol" field
heart.fbs.mean() # Mean of "fbs" field
heart.restecg.mean() # Mean of "restecg" field
heart.thalach.mean() # Mean of "thalach" field
heart.exang.mean() # Mean of "exang" field
heart.oldpeak.mean() # Mean of "oldpeak" field
heart.slope.mean() # Mean of "slope" field
heart.ca.mean() # Mean of "ca" field
heart.thal.mean() # Mean of "thal" field
heart.target.mean() # Mean of "target" field

# to display mode of the fields in data
heart.age.mode() 
heart.sex.mode()
heart.cp.mode()
heart.trestbps.mode()
heart.chol.mode()
heart.fbs.mode()
heart.restecg.mode()
heart.thalach.mode()
heart.exang.mode()
heart.oldpeak.mode()
heart.slope.mode()
heart.ca.mode()
heart.thal.mode()
heart.target.mode()

# to display median of the field
heart.age.median()
heart.trestbps.median()
heart.chol.median()
heart.thalach.median()
heart.oldpeak.median()

# Categorical data in pandas is used for data that takes on a limited and usually fixed number of possible values (categories).

# from 1st moment we determine the representative value of the data
# using mean, median and mode

                           # 2nd Moment
                           
# to determine Variance, Standard Deviation and Range of the data
heart.age.var()
heart.age.std()
range = max(heart.age) - min(heart.age)
range

heart.sex.var()
heart.sex.std()
range = max(heart.sex) - min(heart.sex)
range

heart.cp.var()
heart.cp.std()
range = max(heart.cp) - min(heart.cp)
range

heart.trestbps.var()
heart.trestbps.std()
range = max(heart.trestbps) - min(heart.trestbps)
range

heart.chol.var()
heart.chol.std()
range = max(heart.chol) - min(heart.chol)
range

heart.fbs.var()
heart.fbs.std()
range = max(heart.fbs) - min(heart.fbs)
range

heart.restecg.var()
heart.restecg.std()
range = max(heart.restecg) - min(heart.restecg)
range

heart.thalach.var()
heart.thalach.std()
range = max(heart.thalach) - min(heart.thalach)
range

heart.exang.var()
heart.exang.std()
range = max(heart.exang) - min(heart.exang)
range

heart.oldpeak.var()
heart.oldpeak.std()
range = max(heart.oldpeak) - min(heart.oldpeak)
range

heart.slope.var()
heart.slope.std()
range = max(heart.slope) - min(heart.slope)
range

heart.ca.var()
heart.ca.std()
range = max(heart.ca) - min(heart.ca)
range

heart.thal.var()
heart.thal.std()
range = max(heart.thal) - min(heart.thal)
range

heart.target.var()
heart.target.std()
range = max(heart.target) - min(heart.target)
range

"""From 2nd Moment we find the measure of spread from the center 
of the values in the data"""

                        # 3rd Moment
                        
# Skewness
heart.age.skew()       # Negative Skew(Mode > Median > Mean)
heart.sex.skew()       # Negative Skew(Mode > Median > Mean)  
heart.cp.skew()        # Positive Skew(Mode < Median < Mean)
heart.trestbps.skew()  # Positive Skew(Mode < Median < Mean)
heart.chol.skew()      # Positive Skew(Mode < Median < Mean)
heart.fbs.skew()       # Positive Skew(Mode < Median < Mean)
heart.restecg.skew()   # Positive Skew(Mode < Median < Mean)
heart.thalach.skew()   # Negative Skew(Mode > Median > Mean)
heart.exang.skew()     # Positive Skew(Mode < Median < Mean)
heart.oldpeak.skew()   # Positive Skew(Mode < Median < Mean)
heart.slope.skew()     # Negative Skew(Mode > Median > Mean)
heart.ca.skew()        # Positive Skew(Mode < Median < Mean)
heart.thal.skew()      # Negative Skew(Mode > Median > Mean)
heart.target.skew()    # Negative Skew(Mode > Median > Mean) 

# We understand the data distribution through the table

                          # 4th Moment
# Kurtosis
heart.age.kurt()       # Platykurtic         
heart.sex.kurt()       # Platykurtic
heart.cp.kurt()        # Platykurtic
heart.trestbps.kurt()  # Platykurtic
heart.chol.kurt()      # Leptokurtic
heart.fbs.kurt()       # Platykurtic
heart.restecg.kurt()   # Platykurtic
heart.thalach.kurt()   # Platykurtic
heart.exang.kurt()     # Platykurtic
heart.oldpeak.kurt()   # Platykurtic
heart.slope.kurt()     # Platykurtic
heart.ca.kurt()        # Platykurtic
heart.thal.kurt()      # Platykurtic  
heart.target.kurt()    # Platykurtic

# We determine the sharpness of the data distribution curve
# Platykurtic : low kurtosis (thin tails)
# Leptokurtic : high kurtosis (fat tails)

heart.corr()

# Computes the pairwise correlation of columns

#--------------------------------------------------------------------

# Bar Plot   
plt.bar(height = heart.age, x = np.arange(303))

# Histogram   To understand data distribution   
plt.hist(heart.age, edgecolor = 'black')       #Right Skewed
plt.hist(heart.sex, edgecolor = 'black')
plt.hist(heart.cp, edgecolor = 'black')        #Left Skewed
plt.hist(heart.trestbps, edgecolor = 'black')  #Left Skewed
plt.hist(heart.chol, edgecolor = 'black')      #Left Skewed
plt.hist(heart.fbs, edgecolor = 'black')
plt.hist(heart.restecg, edgecolor = 'black')
plt.hist(heart.thalach, edgecolor = 'black')   #Right Skewed
plt.hist(heart.exang, edgecolor = 'black')     #Left Skewed
plt.hist(heart.oldpeak, edgecolor = 'black')   #Left Skewed
plt.hist(heart.slope, edgecolor = 'black')
plt.hist(heart.ca, edgecolor = 'black')        #Left Skewed
plt.hist(heart.thal, edgecolor = 'black')      #Right Skewed
plt.hist(heart.target, edgecolor = 'black')

# Boxplot    Talks about presence of outliers     Univariant plot   Data distribution
plt.figure()
plt.boxplot(heart.age) # zero outliers
plt.boxplot(heart.sex)
plt.boxplot(heart.cp)
plt.boxplot(heart.trestbps)   
plt.boxplot(heart.chol)
plt.boxplot(heart.fbs)
plt.boxplot(heart.restecg)
plt.boxplot(heart.thalach)
plt.boxplot(heart.exang)
plt.boxplot(heart.oldpeak)
plt.boxplot(heart.slope)
plt.boxplot(heart.ca)
plt.boxplot(heart.thal)
plt.boxplot(heart.target)


# Bivariate Visualization
# Scatter plot :  relationship between 2 different fields
# It talks about 1.Direction(+ve, -ve or no direction)
#                2.Strength (strong, moderate or weak)
#                3.Correlation coefficient r 

plt.scatter(x = heart['age'], y = heart['trestbps'], color = 'green')
# the scatter plot shows No relation between the above fields
# Strength between the 2 fields is strong

plt.scatter(x = heart['restecg'], y = heart['sex'], color = 'pink')
# Scatter plot shows the No relation between the above fields
# Strength between the 2 fields is weak

#----------------------------------------------------------------------

#Analysis of Duplicates

#If data contains multiple rows having same values, they need to be removed data processing.


duplicate = heart.duplicated()  # returns boolean series 
duplicate
sum(duplicate)
# parameters
duplicate1 = heart.duplicated(keep = "last") 
duplicate1 # duplicates in data are marked true except the last occurance

duplicate2 = heart.duplicated(keep = 'first') 
# duplicates in data are marked true except the first occurance
duplicate2

duplicate3 = heart.duplicated(keep = False)  
# marks all duplicates as true
duplicate3

sum(duplicate)

# removing duplicates
data1 = heart.drop_duplicates() # returns dataframe without duplicates
data1

#------------------------------------------------------------------

#OUTLIER ANALYSIS

# Outliers are values in data that may exceed the range of that particular column

# Detection of Outliers 
IQR = heart['trestbps'].quantile(0.75) - heart['trestbps'].quantile(0.25)
IQR   

# IQR is used to measure variability by dividing a data set into quartiles.

plt.boxplot(heart.trestbps)
# the above boxplot contains circles at maximum end which represent Outliers

lower_limit = heart['trestbps'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['trestbps'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['trestbps'])
heart_s = winsor_iqr.fit_transform(heart[['trestbps']])

# Winzoriser removes the Outliers in the field by changing the range of upper or lower limit

plt.boxplot(heart_s.trestbps)
# Above plot displays data without Outliers

# Similarly we can remove Outliers of remaining fields which contain Outliers

#2

IQR = heart['chol'].quantile(0.75) - heart['chol'].quantile(0.25)
IQR


lower_limit = heart['chol'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['chol'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['chol'])
heart_s = winsor_iqr.fit_transform(heart[['chol']])

plt.boxplot(heart_s.chol)


#3

IQR = heart['thalach'].quantile(0.75) - heart['thalach'].quantile(0.25)
IQR


lower_limit = heart['thalach'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['thalach'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['thalach'])
heart_s = winsor_iqr.fit_transform(heart[['thalach']])

plt.boxplot(heart_s.thalach)

#4

IQR = heart['oldpeak'].quantile(0.75) - heart['oldpeak'].quantile(0.25)
IQR


lower_limit = heart['oldpeak'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['oldpeak'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['oldpeak'])
heart_s = winsor_iqr.fit_transform(heart[['oldpeak']])

plt.boxplot(heart_s.oldpeak)


#5

IQR = heart['ca'].quantile(0.75) - heart['ca'].quantile(0.25)
IQR


lower_limit = heart['ca'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['ca'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['ca'])
heart_s = winsor_iqr.fit_transform(heart[['ca']])

plt.boxplot(heart_s.ca)

#6

IQR = heart['thal'].quantile(0.75) - heart['thal'].quantile(0.25)
IQR


lower_limit = heart['thal'].quantile(0.25) - (IQR * 1.5)
upper_limit = heart['thal'].quantile(0.75) + (IQR * 1.5)


winsor_iqr = Winsorizer(capping_method = 'iqr',tail = 'both',fold = 1.5, variables = ['thal'])
heart_s = winsor_iqr.fit_transform(heart[['thal']])

plt.boxplot(heart_s.thal)

# There 6 columns which contain Outliers. These Outliers are removed using Winsorizer
# Above are the Boxplots without Outliers.


#----------------------------------------------------------------------------------------

import pandas as pd 

heart = pd.read_csv("heart.csv")
heart.head()  # By default it gives only 5 rows
heart.tail()
heart.info()
heart.describe()  # It describes only the numeric values

                          # Binarization
                          
heart['age_new'] = pd.cut(heart['age'], bins=[min(heart.age), heart.age.mean(), max(heart.age)], labels=["low", "high"])  # Creating a new field called age_new
# Look out for the break up of the categories
heart.age_new.value_counts()  # It prints the values

heart['sex'] = pd.cut(heart['sex'], bins=[min(heart.sex), heart.sex.mean(), max(heart.sex)], labels=["low", "high"])  # Creating a new field called age_new
# Look out for the break up of the categories
heart.sex.value_counts()  # It prints the values

heart['cp_new'] = pd.cut(heart['cp'], bins=[min(heart.cp), heart.cp.mean(), max(heart.cp)], labels=["low", "high"])  # Creating a new field called age_new
# Look out for the break up of the categories
heart.cp_new.value_counts()  # It prints the values

heart['trestbps_new1'] = pd.cut(heart['trestbps'], bins=[min(heart.trestbps), heart.trestbps.mean(), max(heart.trestbps)], include_lowest=True, labels=["low", "high"])
# Look out for the break up of the categories
heart.trestbps_new1.value_counts()

heart['chol_new1'] = pd.cut(heart['chol'], bins=[min(heart.chol), heart.chol.mean(), max(heart.chol)], include_lowest=True, labels=["low", "high"])
heart.chol_new1.value_counts()

# Repeat the same process for other columns...
heart['fbs_new1'] = pd.cut(heart['fbs'], bins=[min(heart.fbs), heart.fbs.mean(), max(heart.fbs)], include_lowest=True, labels=["low", "high"])
heart.fbs_new1.value_counts()

heart['restecg_new1'] = pd.cut(heart['restecg'], bins=[min(heart.restecg), heart.restecg.mean(), max(heart.restecg)], include_lowest=True, labels=["low", "high"])
heart.restecg_new1.value_counts()

heart['thalach_new1'] = pd.cut(heart['thalach'], bins=[min(heart.thalach), heart.thalach.mean(), max(heart.thalach)], include_lowest=True, labels=["low", "high"])
heart.thalach_new1.value_counts()

heart['exang_new1'] = pd.cut(heart['exang'], bins=[min(heart.exang), heart.exang.mean(), max(heart.exang)], include_lowest=True, labels=["low", "high"])
heart.exang_new1.value_counts()

heart['oldpeak_new1'] = pd.cut(heart['oldpeak'], bins=[min(heart.oldpeak), heart.oldpeak.mean(), max(heart.oldpeak)], include_lowest=True, labels=["low", "high"])
heart.oldpeak_new1.value_counts()

heart['slope_new1'] = pd.cut(heart['slope'], bins=[min(heart.slope), heart.slope.mean(), max(heart.slope)], include_lowest=True, labels=["low", "high"])
heart.slope_new1.value_counts()

heart['ca_new1'] = pd.cut(heart['ca'], bins=[min(heart.ca), heart.ca.mean(), max(heart.ca)], include_lowest=True, labels=["low", "high"])
heart.ca_new1.value_counts()

heart['thal_new1'] = pd.cut(heart['thal'], bins=[min(heart.thal), heart.thal.mean(), max(heart.thal)], include_lowest=True, labels=["low", "high"])
heart.thal_new1.value_counts()

heart['target_new1'] = pd.cut(heart['target'], bins=[min(heart.target), heart.target.mean(), max(heart.target)], include_lowest=True, labels=["low", "high"])
heart.target_new1.value_counts()

                           # Discretization/ Multiple bins

heart['sex_multi'] = pd.cut(heart['sex'], bins=[min(heart.sex), heart.sex.quantile(0.25), heart.sex.mean(), heart.sex.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.sex_multi.value_counts()

# Assuming MaritalDesc is another column in your DataFrame
heart.MaritalDesc.value_counts()

heart['cp_multi'] = pd.cut(heart['cp'], bins=[min(heart.cp), heart.cp.quantile(0.25), heart.cp.mean(), heart.cp.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.cp_multi.value_counts()

heart['trestbps_multi'] = pd.cut(heart['trestbps'], bins=[min(heart.trestbps), heart.trestbps.quantile(0.25), heart.trestbps.mean(), heart.trestbps.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.trestbps_multi.value_counts()

heart['chol_multi'] = pd.cut(heart['chol'], bins=[min(heart.chol), heart.chol.quantile(0.25), heart.chol.mean(), heart.chol.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.chol_multi.value_counts()
heart['fbs_multi'] = pd.cut(heart['fbs'], bins=[min(heart.fbs), heart.fbs.quantile(0.25), heart.fbs.mean(), heart.fbs.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.fbs_multi.value_counts()

# Assuming MaritalDesc is another column in your DataFrame
heart.MaritalDesc.value_counts()

heart['restecg_multi'] = pd.cut(heart['restecg'], bins=[min(heart.restecg), heart.restecg.quantile(0.25), heart.restecg.mean(), heart.restecg.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.restecg_multi.value_counts()

heart['thalach_multi'] = pd.cut(heart['thalach'], bins=[min(heart.thalach), heart.thalach.quantile(0.25), heart.thalach.mean(), heart.thalach.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.thalach_multi.value_counts()

heart['exang_multi'] = pd.cut(heart['exang'], bins=[min(heart.exang), heart.exang.quantile(0.25), heart.exang.mean(), heart.exang.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.exang_multi.value_counts()

heart['oldpeak_multi'] = pd.cut(heart['oldpeak'], bins=[min(heart.oldpeak), heart.oldpeak.quantile(0.25), heart.oldpeak.mean(), heart.oldpeak.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.oldpeak_multi.value_counts()

# Assuming MaritalDesc is another column in your DataFrame
heart.MaritalDesc.value_counts()

heart['slope_multi'] = pd.cut(heart['slope'], bins=[min(heart.slope), heart.slope.quantile(0.25), heart.slope.mean(), heart.slope.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.slope_multi.value_counts()

heart['ca_multi'] = pd.cut(heart['ca'], bins=[min(heart.ca), heart.ca.quantile(0.25), heart.ca.mean(), heart.ca.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.ca_multi.value_counts()

heart['thal_multi'] = pd.cut(heart['thal'], bins=[min(heart.thal), heart.thal.quantile(0.25), heart.thal.mean(), heart.thal.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.thal_multi.value_counts()

heart['target_multi'] = pd.cut(heart['target'], bins=[min(heart.target), heart.target.quantile(0.25), heart.target.mean(), heart.target.quantile(0.75), max(heart.age)], include_lowest=True, labels=["p1", "p2", "p3", "p4"])
heart.target_multi.value_counts()



# We use dummy variables to convert non numeric data to numerica data
# Types of encoding       1. 1Hot encoding         2. Label encoding


# Importing necessary libraries
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Reading the dataset
heart = pd.read_csv(r"/Users/Desktop/heart.csv")

# Dropping unwanted columns from the dataset
heart.drop(['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'], axis=1, inplace=True)

# One-Hot Encoding
# Initializing OneHotEncoder
enc = OneHotEncoder()

# Performing one-hot encoding on the selected columns (starting from the 3rd column)
enc_heart = pd.DataFrame(enc.fit_transform(heart.iloc[:,2:]).toarray())

# Reading the dataset
heart = pd.read_csv(r"/Users/Desktop/heart.csv")

# Rearranging the columns in the original dataframe (if necessary)
heart = heart[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]

# Label Encoding

# Reading the dataset
heart = pd.read_csv(r"/Users/Desktop/heart.csv")
# Initializing LabelEncoder
labelEncoder = LabelEncoder()

# Selecting columns for label encoding (assuming first 9 columns)
x = heart.iloc[:,:9] 








