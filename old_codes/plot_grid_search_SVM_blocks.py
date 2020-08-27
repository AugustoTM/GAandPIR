"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from import_all_data import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print(__doc__)

# Loading dataset
file = "C:/Users/atmiy/Documents/ISAE SUPAERO/2A/S4/PIR/Project_Codes/all_Data.csv"
alldataset = import_alldata(file)
alldataset = normalizingOffData(alldataset)

subdataset = alldataset.loc[:, ['Participant',
                                'Difficulty', 
                                'Section', 
                                'Accuracy',
                                'Off_mean_RR',
                                'Off_std_RR',
                                'Off_mean_HRV',
                                'RR_Max',
                                'RR_Mean',
                                'RR_SD',
                                'RR_normMax',
                                'RR_normMean',
                                'RR_normSD',
                                'RR_SDMean',
                                'HRV7_EndVal',
                                'HRV7_Mean',
                                'HRV7_normEndVal',
                                'HRV7_normMean',
                                'HRV7_normSlope',
                                'HRV20_EndVal',
                                'HRV20_Mean',
                                'HRV20_normEndVal',
                                'HRV20_normMean',
                                'HRV20_Slope',
                                'HRV20_normSlope']]   

### mean offline
subdataset_blocks = subdataset.loc[(subdataset['Section']=='Blocks')&
                                   (subdataset['Difficulty']!='3')]
subdataset_blocks = subdataset_blocks.loc[:, ['Difficulty', 'Section', 'Off_mean_RR', 'Off_std_RR', 
                                              'Off_mean_HRV', 'RR_normMean', 'RR_normSD', 
                                              'RR_normMean', 'HRV7_normMean', 
                                              'RR_normMean', 'HRV7_normEndVal',
                                              'RR_normMean', 'HRV7_normSlope']].reindex()
onlyvalues = subdataset_blocks.values
# Labels
Y = onlyvalues[:,0]
Xoff = onlyvalues[:,2:4]
Xoffscaled = preprocessing.scale(Xoff)
min_max_scaler = preprocessing.MinMaxScaler()
Xoffminmax = min_max_scaler.fit_transform(Xoff)

### online normRRmean and normRRSD
Xon = onlyvalues[:,5:7]
Xonscaled = preprocessing.scale(Xon)
Xonminmax = min_max_scaler.fit_transform(Xon)

### online normRRmean and normHRV7
Xnorm = onlyvalues[:,7:9]
Xnormscaled = preprocessing.scale(Xnorm)
Xnormminmax = min_max_scaler.fit_transform(Xnorm)

#### online RRmean and normHRV7EndVal
Xend = onlyvalues[:,9:11]
Xendscaled = preprocessing.scale(Xend)
Xendminmax = min_max_scaler.fit_transform(Xend)

#### online RRmean and normHRV7EndVal
Xslope = onlyvalues[:,11:]
Xslopescaled = preprocessing.scale(Xslope)
Xslopeminmax = min_max_scaler.fit_transform(Xslope)

####  OFFLINE

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    Xoffscaled, Y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],
                     'C': [1, 1.5, 2, 2.5]}]

scores = ['precision_macro', 'recall_macro', 'f1_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# output model cross-validation
best_param = clf.best_params_
clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, Xoffscaled, Y, scoring=scoring, cv=5, return_train_score=False)
print(sorted(scores.keys()))
print("Precision")
print(np.mean(scores['test_prec_macro']))
print("Recall")
print(np.mean(scores['test_rec_macro']))
print("F1-score")
print(np.mean(scores['test_f1_macro']))

print("Confusion Matrix")
predicted = cross_val_predict(clf, Xoffscaled, Y, cv=5)
print(confusion_matrix(Y, predicted))
tn, fp, fn, tp = confusion_matrix(Y, predicted).ravel()
print("Specificity")
print(str(tn/(tn+fp)))
print("Sensitivity")
print(str(tp/(tp+fn)))


####################
### RRnorm and EndVal

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    Xendscaled, Y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],
                     'C': [1, 1.5, 2, 2.5]}]

scores = ['precision_macro', 'recall_macro', 'f1_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# output model cross-validation
best_param = clf.best_params_
clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, Xendscaled, Y, scoring=scoring, cv=5, return_train_score=False)
print(sorted(scores.keys()))
print("Precision")
print(np.mean(scores['test_prec_macro']))
print("Recall")
print(np.mean(scores['test_rec_macro']))
print("F1-score")
print(np.mean(scores['test_f1_macro']))

print("Confusion Matrix")
predicted = cross_val_predict(clf, Xendscaled, Y, cv=5)
print(confusion_matrix(Y, predicted))
tn, fp, fn, tp = confusion_matrix(Y, predicted).ravel()
print("Specificity")
print(str(tn/(tn+fp)))
print("Sensitivity")
print(str(tp/(tp+fn)))

#######################
#### RRmean and RR_SD slope
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    Xslopescaled, Y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],
                     'C': [1, 1.5, 2, 2.5]}]

scores = ['precision_macro', 'recall_macro', 'f1_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# output model cross-validation
best_param = clf.best_params_
clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, Xslopescaled, Y, scoring=scoring, cv=5, return_train_score=False)
print(sorted(scores.keys()))
print("Precision")
print(np.mean(scores['test_prec_macro']))
print("Recall")
print(np.mean(scores['test_rec_macro']))
print("F1-score")
print(np.mean(scores['test_f1_macro']))

print("Confusion Matrix")
predicted = cross_val_predict(clf, Xslopescaled, Y, cv=5)
print(confusion_matrix(Y, predicted))
tn, fp, fn, tp = confusion_matrix(Y, predicted).ravel()
print("Specificity")
print(str(tn/(tn+fp)))
print("Sensitivity")
print(str(tp/(tp+fn)))


#######################
#### RRmean and RR_SD classique window
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    Xonscaled, Y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],
                     'C': [1, 1.5, 2, 2.5]}]

scores = ['precision_macro', 'recall_macro', 'f1_macro']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# output model cross-validation
best_param = clf.best_params_
clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, Xonscaled, Y, scoring=scoring, cv=5, return_train_score=False)
print(sorted(scores.keys()))
print("Precision")
print(np.mean(scores['test_prec_macro']))
print("Recall")
print(np.mean(scores['test_rec_macro']))
print("F1-score")
print(np.mean(scores['test_f1_macro']))

print("Confusion Matrix")
predicted = cross_val_predict(clf, Xonscaled, Y, cv=5)
print(confusion_matrix(Y, predicted))
tn, fp, fn, tp = confusion_matrix(Y, predicted).ravel()
print("Specificity")
print(str(tn/(tn+fp)))
print("Sensitivity")
print(str(tp/(tp+fn)))
