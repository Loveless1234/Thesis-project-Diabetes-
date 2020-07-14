

# Python libraries
# Classic,data manipulation and linear algebra
import pandas as pd
import numpy as np
import pickle



# Data processing, metrics and modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


#changing the column names
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','diabetes']
data=pd.read_csv("/Users/ranjitsah/Documents/Data science/ML/projekt_thesis/diabetes.csv",header=0,names=col_names)



data.shape
data.describe().T

# # Data Preparation(Missing values)

zeroCols = ['glucose', 'bp', 'skin', 'insulin','bmi'] # Columns with incorrect Zero values 
df2 = data.copy() # create a copy of the original dataframe 
df2[zeroCols] = df2[zeroCols].replace(0,np.NaN) #Replace 0s with NaNs
df2.head()


df2.isnull().sum()/len(df2)*100

dataset=df2.copy()


# # Replacing  Missing Values and EDA

# function To Fill Missing values
def median_target(var):   
    temp = df2[df2[var].notnull()]
    temp = temp[[var, 'diabetes']].groupby(['diabetes'])[[var]].median().reset_index()
    return temp


# # Insulin
median_target('insulin')

df2.loc[(df2['diabetes'] == 0 ) & (df2['insulin'].isnull()), 'insulin'] = 102.5 #+ np.random.normal(0,1,236)
df2.loc[(df2['diabetes'] == 1 ) & (df2['insulin'].isnull()), 'insulin'] = 169.5 #+ np.random.normal(0,1,138)
#(236,138)


# # Glucose
median_target('glucose')

df2.loc[(df2['diabetes'] == 0 ) & (df2['glucose'].isnull()), 'glucose'] = 107.0
df2.loc[(df2['diabetes'] == 1 ) & (df2['glucose'].isnull()), 'glucose'] = 140.0


# # Skin
median_target('skin')


df2.loc[(df2['diabetes'] == 0 ) & (df2['skin'].isnull()), 'skin'] = 27.0 #+ np.random.normal(0,1,139)
df2.loc[(df2['diabetes'] == 1 ) & (df2['skin'].isnull()), 'skin'] = 32.0 #+ np.random.normal(0,1,88)



# # BP
median_target('bp')

df2.loc[(df2['diabetes'] == 0 ) & (df2['bp'].isnull()), 'bp'] = 70.0 #+ np.random.normal(0,1,19)
df2.loc[(df2['diabetes'] == 1 ) & (df2['bp'].isnull()), 'bp'] = 74.0 #+ np.random.normal(0,1,16)


# # BMI
median_target('bmi')

df2.loc[(data['diabetes'] == 0 ) & (df2['bmi'].isnull()), 'bmi'] = 30.1
df2.loc[(data['diabetes'] == 1 ) & (df2['bmi'].isnull()), 'bmi'] = 34.3


# # creating Decision Tree Model

array=df2.values
feature_cols=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age']

#split dataset in features and target variable
X=array[:,0:8]
y=array[:,8]

X[0]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?
print(round(accuracy_score(y_test,y_pred)*100,2))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred) )#output_dict=True




classifier = RandomForestClassifier(n_estimators=200,random_state=7)
classifier = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
classifier = pickle.load(open('model.pkl','rb'))


print(classifier.predict([[7,136,74,26,135,26,0.647,50]]))
