__author__ = 'masai'
# This code is to build classifier for student Y1 test performance
# Features are demographics, att, discipline, quarter performance

import math
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, expon
from sklearn import cross_validation, metrics, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import operator

# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------
def load_data(file_path):
    fea_name = ['AttPer',    'Suspension',   'Mobility',     'Male',         'White',
                'Black',     'Asian',        'OtherRace',    'ELL',          'Urban',
                'HighNeeds', 'FreeLunch',    'ReducedLunch', 'SpecialEd',    'FirstLangEng',
                'McasElaP',  'McasElaNI',    'McasElaW',	 'McasMathP',    'McasMathNI',
                'McasMathW', 'Discipline',   'M1Course',     'Q1Course',     'Q2Course',
                'Q3Course',  'Q4Course',     'Y1Fail']   # feature names, the last one is y
    df = pd.read_csv(file_path, sep=',', names=fea_name, header=0)  # read csv, the first row is header

    # get X and y
    all_fea_idx = [i for i in range(0, (len(fea_name) - 1))]    # index for all features
    drop_fea_idx = []   # the index of features to be dropped
    kept_fea_idx = [i for i in all_fea_idx if i not in drop_fea_idx]    # index of kept features
    X = df.values[:, kept_fea_idx]   # feature set
    y = df.values[:, -1]    # class labels, where 1 for Y1=Fail
    return X, y, fea_name

# ----------------------------------------------------------------------------------------------------------------------
# feature selection
# ----------------------------------------------------------------------------------------------------------------------
def train_feature_selection(X, y, fname):
    p = 60    # keep
    print('Performing feature selection:', p, '% best')
    selector = GenericUnivariateSelect(f_classif, mode='percentile', param=p)
    X_new = selector.fit_transform(X, y)
    idx = list(selector.get_support(indices=True))
    fname_new = [None] * (len(idx) + 1)
    c = 0
    for i in idx:
        fname_new[c] = fname[i]
        c += 1
    p_f = {}
    for i in range(len(fname_new) - 1):
        p_f[fname_new[i]] = selector.pvalues_[idx[i]] # round(-np.log10(selector.pvalues_[idx[i]]), 1)
    print('     p_values for selected features: ')
    for i in p_f:
        print('         ', i, ':', p_f[i])
    fname_new[c] = fname[-1]
    return X_new, y, fname_new
    # train_cross_validation(X_new, y, fname_new)

# ----------------------------------------------------------------------------------------------------------------------
# Perform a search for optimal parameters of classifier
# ----------------------------------------------------------------------------------------------------------------------
def train_parameter_search(X, y):
    print('Performing parameter search:')
    # Classifiers and parameters
    clf_svm = svm.SVC()
    svm_param_grid = {'kernel': ['rbf']
                      , 'gamma': [0.05, 0.1, 0.5]
                      , 'C': [1, 2, 10, 100]
                      , 'class_weight': ['auto', None]
                      }
    svm_param_dist = {'kernel': ['rbf']
                      , 'gamma': expon(scale=.1)
                      , 'C': expon(scale=100)
                      , 'class_weight': ['auto', None]
                      }
    clf_rf = RandomForestClassifier()
    rf_param_grid = {'n_estimators': [20, 50]
                     , 'criterion': ['entropy', 'gini']
                     , 'max_features': [1, 'sqrt', 'log2']
                     , 'max_depth': [3, 5]
                     #, 'bootstrap': [True, False],
                     #, 'class_weight': ['auto', 'subsample', None]
                     }
    rf_param_dist = {'n_estimators': randint(20, 150)
                     , 'criterion': ['entropy', 'gini']
                     , 'max_features': randint(1, 6)
                     , 'max_depth': [3, 5, None]
                     #, 'min_samples_split': randint(1, 11)
                     #, 'min_samples_leaf': randint(1, 11)
                     , 'bootstrap': [True, False]
                     , 'class_weight': ['auto', 'subsample', None]
                     }
    # clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10))
    # ada_param_grid = {'n_estimators': [20, 50]
    #                   , 'learning_rate': [1, 1.5, 2]
    #                   , 'algorithm': ['SAMME']
    #                   }
    # ada_param_dist = {'n_estimators': randint(20, 150)
    #                   , 'learning_rate': [1, 1.5, 2]
    #                   , 'algorithm': ['SAMME']
    #                  }

    # Perform search for optimal parameters
    search_grid_svm = GridSearchCV(clf_svm, param_grid=svm_param_grid, cv=3, scoring='recall_weighted').fit(X, y)
    search_rand_svm = RandomizedSearchCV(clf_svm, param_distributions=svm_param_dist, n_iter=10).fit(X, y)
    search_grid_rf = GridSearchCV(clf_rf, param_grid=rf_param_grid, cv=3, scoring='recall_weighted').fit(X, y)
    search_rand_rf = RandomizedSearchCV(clf_rf, param_distributions=rf_param_dist, n_iter=10).fit(X, y)
    # search_grid_ada = GridSearchCV(clf_ada, param_grid=ada_param_grid, cv=3, scoring='recall_weighted').fit(X, y)
    # search_rand_ada = RandomizedSearchCV(clf_ada, param_distributions=ada_param_dist, n_iter=10).fit(X, y)

    # Return best combination
    best_scores = {'search_grid_svm':search_grid_svm.best_score_
                   ,'search_rand_svm':search_rand_svm.best_score_
                   ,'search_grid_rf':search_grid_rf.best_score_
                   ,'search_rand_rf':search_rand_rf.best_score_
                   # ,'search_grid_ada':search_grid_ada.best_score_
                   # ,'search_rand_ada':search_rand_ada.best_score_
                   }
    best_clf = max(best_scores, key=best_scores.get)
    print('     Best classifier and parameters:')
    if best_clf == 'search_grid_svm':
        print('         SVM', search_grid_svm.best_params_)
        return 'svm', search_grid_svm.best_params_
    elif best_clf == 'search_rand_svm':
        print('         SVM', search_rand_svm.best_params_)
        return 'svm', search_rand_svm.best_params_
    elif best_clf == 'search_grid_rf':
        print('         Random Forest', search_grid_rf.best_params_)
        return 'rf', search_grid_rf.best_params_
    elif best_clf == 'search_rand_rf':
        print('         Random Forest', search_rand_rf.best_params_)
        return 'rf', search_rand_rf.best_params_
    # elif best_clf == 'search_grid_ada':
    #     print('         AdaBoost', search_grid_ada.best_params_)
    #     return 'ada', search_grid_ada.best_params_
    # elif best_clf == 'search_rand_ada':
    #     print('         AdaBoost', search_rand_ada.best_params_)
    #     return 'ada', search_rand_ada.best_params_

# ----------------------------------------------------------------------------------------------------------------------
# Perform Cross validation
# ----------------------------------------------------------------------------------------------------------------------
def train_cross_validation(X, y, par, which_classifier):
    nfold = 10  # number of folds
    print('Performing %i-fold cross-validation on training data:' % nfold)
    # # Classification
    if which_classifier == 'svm':
        if bool(par):
            clf = svm.SVC(C=par.get('C'), gamma=par.get('gamma'), kernel=par.get('kernel'))
            clf.fit(X, y)
        else:
            clf = svm.SVC(C=100.0, gamma=0.1, kernel='rbf')
            clf.fit(X, y)
    elif which_classifier == 'rf':  # Random Forest
        if bool(par):
            clf = RandomForestClassifier(n_estimators=par.get('n_estimators'),
                                         criterion=par.get('criterion'),
                                         max_features=par.get('max_features'),
                                         max_depth=par.get('max_depth'),
                                         bootstrap=par.get('bootstrap'),
                                         class_weight=par.get('class_weight'))
            clf.fit(X, y)
        else:
            clf = RandomForestClassifier(n_estimators=100,
                                         criterion='gini',
                                         max_features=1,
                                         max_depth=5,
                                         bootstrap=False,
                                         class_weight=None)
            clf.fit(X, y)
    elif which_classifier == 'adaboost':
        if bool(par):
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                                     n_estimators=par.get('n_estimators'),
                                     learning_rate=par.get('learning_rate'),
                                     algorithm=par.get('algorithm'))
            clf.fit(X, y)
        else:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=200, learning_rate=1.5,
                                     algorithm="SAMME")
            clf.fit(X, y)
    scores = cross_validation.cross_val_score(clf, X, y, cv=nfold)
    print('     Classifier:', clf)
    print('     Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return clf

# ----------------------------------------------------------------------------------------------------------------------
# Perform classification for training data
# ----------------------------------------------------------------------------------------------------------------------
def train_classification(X, y):
    clf = RandomForestClassifier(n_estimators=50  # number of trees in forest
                                 , criterion='entropy'  # measure the quality of a split, 'gini' or 'entropy'
                                 , max_features='sqrt'  # number of features to consider when looking for the best split
                                 , max_depth=3  # maximum depth of tree
                                 , bootstrap=True  # whether bootstrap samples when building trees
                                 # ,class_weight={0: 1, 1: 2}  # “auto” mode adjusts the weights inversely proportional to class frequencies in the input data
                                 )
    clf.fit(X, y)
    print('Performing RF classification:')
    y_pred = clf.predict(X)     # training prediction
    print('     Confusion Matrix: \n', metrics.confusion_matrix(y, y_pred))
    print('     Scores: \n', metrics.classification_report(y, y_pred))
    return clf

# ----------------------------------------------------------------------------------------------------------------------
# Perform a transform for testing data based on training
# ----------------------------------------------------------------------------------------------------------------------
def feature_fit(X, fname_train, fname_test):
    if set(fname_train) == set(fname_test):
        return X
    else:
        idx = []
        for i in range(len(fname_train) - 1):
            idx.append(fname_test.index(fname_train[i]))
        X = np.array(X)
        X = X[:, idx]
        return X

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    X_train, y_train, fea_name = load_data('TrainingData_Students.csv')
    print('Make your choice: 0 for no, 1 for yes')
    feature_selection = int(input('     Feature Selection? '))
    parameter_search = int(input('     Parameter Selection? '))
    cross_validation = int(input('     Cross Validation? '))

    # perform the selected step
    if feature_selection:
        X_train, y_train, fea_name = train_feature_selection(X_train, y_train, fea_name)

    # # Search Optimal Parameters
    par = {}
    which_classifier = 'rf'
    if parameter_search:
        which_classifier, par = train_parameter_search(X_train, y_train)

    if cross_validation:
        clf = train_cross_validation(X_train, y_train, par, which_classifier)
    else:
        clf = train_classification(X_train, y_train)

    print('Performing prediction for testing data')
    X_test, y_test, f_name = load_data('TestingData_Students.csv')
    X_test = feature_fit(X_test, fea_name, f_name)
    print(X_test.shape)
    y_test_pred = clf.predict(X_test)
    # print('     Confusion Matrix: \n', metrics.confusion_matrix(y_test, y_test_pred))
    # print('     Scores: \n', metrics.classification_report(y_test, y_test_pred))
    print('     Accuracy: ', metrics.accuracy_score(y_test, y_test_pred))


if __name__ == "__main__":
    main()
