

import numpy as np
import pandas as pd
import seaborn as sns

# Reading the dataset

url = "https://raw.github.com/mattdelhey/kaggle-titanic/master/Data/train.csv"
titanic = pd.read_csv(url) 
print(titanic.head(10))

# Data Analysis

titanic.shape

#There are **891** rows & **11 columns**. This means there are 891 datapoints in the dataset & 11 features."""

titanic.columns

#Out of these features, the feature **'survived' is the target feature**. """

titanic.info()

#There are **5 object fields** which needs to be encoded. 

#'age', 'cabin' & embarked has some **missing values**


#So I need to know how many Nan values are there in each columns.

titanic.isna().sum()

# Data Visualization

import matplotlib.pyplot as plt


plt.figure(figsize=(10,10))
sns.heatmap(titanic.corr(), annot=True, linewidths=0.5, fmt= '.3f')

titanic.corr()

#By the previous knowledge we have, let's create a new feature telling **whether the passenger is man, woman or a child.**"""

def woman_child_or_man(passenger):
    age, sex = passenger
    if age < 16:
        return "child"
    else:
        return dict(male="man", female="woman")[sex]

titanic["who"] = titanic[["age", "sex"]].apply(woman_child_or_man, axis=1)
titanic.head()

#We will create another feature to see wether a person was an adult male or not."""

titanic["adult_male"] = titanic.who == "man"
titanic.head()

#We can have another feature with the deck information."""

titanic["deck"] = titanic.cabin.str[0]
titanic.head()

#Now one more feature can be created, whether the passenger was alone or not. So let's do this."""

titanic["alone"] = ~(titanic.parch + titanic.sibsp).astype(bool)
titanic.head()

#Now let's try to look at the trends in different feature."""

sns.factorplot("pclass", "survived", data=titanic).set(ylim=(0, 1))

#From here we see that if a passenger travelled in 1st class, the survival rate is highest and equal to 0.63. If a passenger travelled in 2nd class, the survival rate is medium and equal to 0.5. If a passenger travelled in 3rd class, the survival rate is lowest and equal to 0.3

#Let's see how the above case is dependent on the **sex of the passenger.**


sns.factorplot("pclass", "survived", data=titanic, hue="sex")

#It;s pretty clear that the surviavl of female passengers is much more than the male passengers. From here we see that if a passenger travelled in 1st class and was female then their survival chance is most. On the other hand, if a passenger travelled in 3rd class amd was male then their survival chance is least. So we can combine these two features to **create new feature**.

#Let's have a similar observation with the features **'class' & 'who'**


sns.factorplot("pclass", "survived", data=titanic, hue="who")

#From here also we can have similar observation. We get 9 cases from here and we will be building a feature based on it in a while.

#Let's try to find the trends with **the feature 'alone' & 'adult_male'**.


sns.factorplot("alone", "survived", data=titanic, hue="sex")

sns.factorplot("adult_male", "survived", data=titanic, hue="sex")

#Now let's see what effect does the feature **'deck'** has."""

sns.barplot("deck", "survived", data=titanic,order=['A','B','C','D','E','F','G'])

#Now let's try to combine 3 features together."""

sns.factorplot("alone", "survived", data=titanic, hue="sex",col="pclass")

# Data Preprocessing

#Let's have the object fields encoded.


#encoding deck

dk = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
titanic['deck']=titanic.deck.map(dk)
titanic.head()

# encoding embarked


titanic['embarked'].value_counts()

e = {'S':3,'Q':2, 'C':1}
titanic['embarked']=titanic.embarked.map(e)
titanic.head()

# encoding gender

genders = {"male": 0, "female": 1}
titanic['sex'] = titanic['sex'].map(genders)
titanic.head()

#encoding who

wh = {'child':3,'woman':2, 'man':1}
titanic['who']=titanic.who.map(wh)

titanic.head()

#Now we need to impute the **Missing Values**

#There are alot of missing values in deck. So we will simply fill it with **0**


#imputing deck
titanic['deck']=titanic['deck'].fillna(0)
titanic.head()

#There are only 2 missing vaues in 'embarked'. So we will find out which of the values in embarked has **maximum occurence** and fill the missing values with **that value**."""

#imputing embarked

titanic['embarked'].value_counts()

titanic['embarked']=titanic['embarked'].fillna('3.0')
titanic.head()

#Now we will impute the missing values in **'age'**."""

#imputing age

m=titanic['age'].mean()
#m

titanic['age']=titanic['age'].fillna(m)
titanic.head()

# Adding New Features"""

def process_family(parameters):
    x,y=parameters
    # introducing a new feature : the size of families (including the passenger)
    family_size = x+ y + 1
    if (family_size==1):
      return 1 # for singleton
    elif(2<= family_size <= 4 ):
      return 2 #for small family
    else:
      return 3 #for big family

titanic['FAM_SIZE']= titanic[['parch','sibsp']].apply(process_family, axis=1)
titanic.head()

# to get title from the name.

titles = set()
for name in titanic['name']:
  titles.add(name.split(',')[1].split('.')[0].strip())

titles #all the salutations present in my dataset.

len(titles)

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles():
  # we extract the title from each name
  titanic['title'] = titanic['name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
  # a map of more aggregated title
  # we map each title
  titanic['title'] = titanic.title.map(Title_Dictionary)
  return titanic

titanic = get_titles()
titanic.head()
#titanic.to_csv('titanic_df_title.csv')
#Now we need to encode these titles. Right now I will use one-hot encoding with this."""

titles_dummies = pd.get_dummies(titanic['title'], prefix='title')
titles_dummies_dict = dict(titles_dummies.loc[0])
print(titles_dummies_dict)
titanic = pd.concat([titanic, titles_dummies], axis=1)
titanic.head()

#And finally the Feature that we observed during the visualization."""

def new_fe(parameters):
  p,w=parameters
  
  if (p==1):
    if (w==1):
      return 1
    elif (w==2):
      return 2
    elif (w==3):
      return 3
  elif (p==2):
    if (w==1):
      return 4
    elif (w==2):
      return 5
    elif (w==3):
      return 6
  elif (p==3):
    if (w==1):
      return 7
    elif (w==2):
      return 8
    elif (w==3):
      return 9

titanic['pcl_wh']= titanic[['pclass','who']].apply(new_fe, axis=1)
titanic.head()

#Now we will drop all the features which I don't want."""

titanic.columns

drop_list=['name','ticket','fare', 'cabin','title']
titanic = titanic.drop(drop_list, axis=1)
titanic.head()

plt.figure(figsize=(20,20))
sns.heatmap(titanic.corr(), annot=True, linewidths=0.5, fmt= '.3f')

# Build the Models

#The first task will be to **split the dataset** into train set and test set.


X_train = titanic.drop("survived", axis=1)
Y_train = titanic["survived"]

from sklearn.model_selection import train_test_split

# splitting data in training set(70%) and test set(30%).
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

"""## Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() #create the object of the model
lr = lr.fit(x_train,y_train)

#lr.predict()
#input = [2,1,14.000000,1,0,1,3,False,0.0,False,2,0,0,0,1,0,0,6]
#inp_nparray = np.array(input)
#type(inp_nparray)
#lr.predict([inp_nparray])

#dumping the model
import pickle
pickle.dump(lr,open('model_titanic.pkl','wb'))

'''from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score

act = accuracy_score(y_train,lr.predict(x_train))
print('Training Accuracy is: ',(act*100))
p = precision_score(y_train,lr.predict(x_train))
print('Training Precision is: ',(p*100))
r = recall_score(y_train,lr.predict(x_train))
print('Training Recall is: ',(r*100))
f = f1_score(y_train,lr.predict(x_train))
print('Training F1 Score is: ',(f*100))

act = accuracy_score(y_test,lr.predict(x_test))
print('Test Accuracy is: ',(act*100))
p = precision_score(y_test,lr.predict(x_test))
print('Test Precision is: ',(p*100))
r = recall_score(y_test,lr.predict(x_test))
print('Test Recall is: ',(r*100))
f = f1_score(y_test,lr.predict(x_test))
print('Test F1 Score is: ',(f*100))

'''
'''
## Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 3, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features=0.5, 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)
rf = rf.fit(x_train,y_train)

act = accuracy_score(y_train,rf.predict(x_train))
print('Training Accuracy is: ',(act*100))
p = precision_score(y_train,rf.predict(x_train))
print('Training Precision is: ',(p*100))
r = recall_score(y_train,rf.predict(x_train))
print('Training Recall is: ',(r*100))
f = f1_score(y_train,rf.predict(x_train))
print('Training F1 Score is: ',(f*100))

act = accuracy_score(y_test,rf.predict(x_test))
print('Test Accuracy is: ',(act*100))
p = precision_score(y_test,rf.predict(x_test))
print('Test Precision is: ',(p*100))
r = recall_score(y_test,rf.predict(x_test))
print('Test Recall is: ',(r*100))
f = f1_score(y_test,rf.predict(x_test))
print('Test F1 Score is: ',(f*100))

"""## Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier


dt = DecisionTreeClassifier()
dt=dt.fit(x_train, y_train)

act = accuracy_score(y_train,dt.predict(x_train))
print('Training Accuracy is: ',(act*100))
p = precision_score(y_train,dt.predict(x_train))
print('Training Precision is: ',(p*100))
r = recall_score(y_train,dt.predict(x_train))
print('Training Recall is: ',(r*100))
f = f1_score(y_train,dt.predict(x_train))
print('Training F1 Score is: ',(f*100))

act = accuracy_score(y_test,dt.predict(x_test))
print('Test Accuracy is: ',(act*100))
p = precision_score(y_test,dt.predict(x_test))
print('Test Precision is: ',(p*100))
r = recall_score(y_test,dt.predict(x_test))
print('Test Recall is: ',(r*100))
f = f1_score(y_test,dt.predict(x_test))
print('Test F1 Score is: ',(f*100))

#Since Random Forest Classifier performs the best, so that will be chosen as the final model."""
'''