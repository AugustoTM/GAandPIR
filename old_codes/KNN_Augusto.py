# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:49:30 2019

@author: atmiy
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd

bankdata = pd.read_csv("C:/Users/atmiy/Documents/ISAE SUPAERO/2A/S4/PIR/Project_Codes/results4.csv")
#bankdata = pd.read_csv("C:/Users/atmiy/Documents/ISAE SUPAERO/2A/S4/PIR/Project_Codes/Individual4_sujet21.csv")

bankdata.shape
bankdata.head()

X = bankdata.drop('Level', axis=1)  
y = bankdata['Level']

iteration = 50
vecmean = [None]*iteration
for c in range (0,iteration):
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train, y_train)  

    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import precision_score
    from sklearn.metrics import classification_report, confusion_matrix  
    #print(confusion_matrix(y_test, y_pred))  
    #print('\n')
    #print(classification_report(y_test, y_pred)) 
    
    from sklearn.metrics import accuracy_score
    vecmean[c] = accuracy_score(y_test,y_pred)
import statistics
oi = statistics.mean(vecmean)
tchau = statistics.stdev(vecmean)
print(statistics.mean(vecmean))
print(statistics.stdev(vecmean))