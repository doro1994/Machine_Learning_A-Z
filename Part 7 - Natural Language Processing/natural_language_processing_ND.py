# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked'].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Importing classifier classes
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Creating classifier objects
classifier_NB = GaussianNB()
classifier_LR = LogisticRegression(solver = 'liblinear')
classifier_SVM = SVC(kernel = 'rbf')
classifier_DT = DecisionTreeClassifier(criterion = 'entropy')
classifier_RF = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
classifier_KN = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

# Training classifiers
classifier_NB.fit(X_train, y_train)
classifier_LR.fit(X_train, y_train)
classifier_SVM.fit(X_train, y_train)
classifier_DT.fit(X_train, y_train)
classifier_RF.fit(X_train, y_train)
classifier_KN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NB = classifier_NB.predict(X_test)
y_pred_LR = classifier_LR.predict(X_test)
y_pred_SVM = classifier_SVM.predict(X_test)
y_pred_DT = classifier_DT.predict(X_test)
y_pred_RF = classifier_RF.predict(X_test)
y_pred_KN = classifier_KN.predict(X_test)

# Making the Confusion Matrix for each classifier
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)
cm_LR = confusion_matrix(y_test, y_pred_LR)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
cm_DT = confusion_matrix(y_test, y_pred_DT)
cm_RF = confusion_matrix(y_test, y_pred_RF)
cm_KN = confusion_matrix(y_test, y_pred_KN)

def Evaluate(cm: type(cm_NB)) -> dict:
    TP = cm[1 ,1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    return {"Accuracy" : Accuracy , 
            "Precision" : Precision, 
            "Recall" : Recall,
            "F1_score" : F1_score}

Evalutaion_NB = Evaluate(cm_NB)
Evalutaion_LR = Evaluate(cm_LR)
Evalutaion_SVM = Evaluate(cm_SVM)
Evalutaion_DT = Evaluate(cm_DT)
Evalutaion_RF = Evaluate(cm_RF)
Evalutaion_KN = Evaluate(cm_KN)





