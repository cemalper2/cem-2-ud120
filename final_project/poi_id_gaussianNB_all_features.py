#!/usr/bin/python

import sys
import pickle
import numpy as np 
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','bonus','exercised_stock_options'] # You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'from_poi_to_this_person','from_this_person_to_poi','poi_ratio', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOWRY CHARLES P', 0)
data_dict.pop('KAMINSKI WINCENTY J', 0)
data_dict.pop('BHATNAGAR SANJAY', 0)
data_dict.pop('LAVORATO JOHN J', 0)



### Task 3: Create new feature(s)
#for person in data_dict:
#    if not (np.isnan(float(data_dict[person]['from_this_person_to_poi'])) or np.isnan(float(data_dict[person]['from_poi_to_this_person']))): 
#        data_dict[person]['total_poi_messages'] = data_dict[person]['from_poi_to_this_person'] + data_dict[person]['from_this_person_to_poi']
#    else:
#        data_dict[person]['total_poi_messages'] = 'NaN'

for person in data_dict:
    if not (np.isnan(float(data_dict[person]['from_this_person_to_poi'])) or np.isnan(float(data_dict[person]['from_poi_to_this_person']))): 
        try:
            data_dict[person]['from_to_ratio'] = float(data_dict[person]['from_poi_to_this_person']) / data_dict[person]['from_this_person_to_poi']
        except: 
            data_dict[person]['from_to_ratio'] = 'NaN'
    else:
        data_dict[person]['from_to_ratio'] = 'NaN'

for person in data_dict:
    if not (np.isnan(float(data_dict[person]['from_this_person_to_poi'])) or np.isnan(float(data_dict[person]['from_poi_to_this_person'])) or np.isnan(float(data_dict[person]['from_messages'])) or np.isnan(float(data_dict[person]['to_messages']))): 
        try:
            data_dict[person]['poi_ratio'] = float(data_dict[person]['from_poi_to_this_person'] +  data_dict[person]['from_this_person_to_poi']) / (data_dict[person]['from_poi_to_this_person'] +  data_dict[person]['from_this_person_to_poi'] + data_dict[person]['to_messages'] +  data_dict[person]['from_messages'])
        except: 
            data_dict[person]['poi_ratio'] = 'NaN'
    else:
        data_dict[person]['poi_ratio'] = 'NaN'





### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#features = preprocessing.scale(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm  import SVC
from sklearn.feature_selection import SelectKBest
from sklearn import cluster
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
clf3 = GaussianNB()
pca = decomposition.PCA(n_components = 9)
selector = SelectPercentile(f_classif, percentile=100)
clf = Pipeline(steps=[('selector',selector), ('scaler',scaler), ('pca', pca), ('k_means', clf3)])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

### K-FOLD X-VALIDATION BEGIN
#kf = KFold(n = len(data), n_folds=5)
#ind = 1
#for train_index, test_index in kf:
#    features_train = np.array(features)[train_index]
#    labels_train = np.array(labels)[train_index]
#    features_test = np.array(features)[test_index]
#    labels_test = np.array(labels)[test_index]
#    print '### K-fold ind %d ###' %(ind)
#    clf.fit(features_train,labels_train)
#    clf2.fit(features_train,labels_train)
#    clf3.fit(features_train,labels_train)
#    print"Naive Bayes without PCA: %0.3f" % (clf.score(features_test,labels_test))
#    print"Decision Tree Classifier without PCA: %0.3f" % (clf2.score(features_test,labels_test))
#    print"SVC without PCA: %0.3f" % (clf3.score(features_test,labels_test))
#    features_train_pca = pca.fit_transform(features_train)
#    features_test_pca = pca.transform(features_test)
#    clf.fit(features_train_pca,labels_train)
#    clf2.fit(features_train_pca,labels_train)
#    clf3.fit(features_train_pca,labels_train)
#    print"Naive Bayes with PCA: %0.3f" % (clf.score(features_test_pca,labels_test))
#    print"Decision Tree Classifier with PCA: %0.3f" % (clf2.score(features_test_pca,labels_test))
#    print"SVC with PCA: %0.3f" % (clf3.score(features_test_pca,labels_test))    
#    print"\n"
#    ind = ind + 1
### K-FOLD X-VALIDATION END
#
    
#print "Fitting the classifier to the training set"
#t0 = time()
#
    
## STRATIFIED TRIAL BEGIN##
#class_0_recall = []
##cv = StratifiedShuffleSplit(labels, 100, random_state = 42, test_size = 0.3)
#cv = StratifiedShuffleSplit(labels, n_iter=1000, random_state=42)
#f1_score_list = []
#for train_idx, test_idx in cv: 
#    features_train = []
#    labels_train = []
#    features_test = []
#    labels_test = []
#    for ii in train_idx:
#        features_train.append( features[ii] )
#        labels_train.append( labels[ii] )
#    for jj in test_idx:
#        features_test.append( features[jj] )
#        labels_test.append( labels[jj] )
#    clf = SVC(kernel = 'rbf', C = 1e7, gamma = 1e-5) # low C: more support, tries to fit more (overfitting), high gamma: points influence region too low (i.e. islands of decision surfaces, overfitting ?)
#    features_train_pca = pca.fit_transform(np.array(features_train))
#    features_test_pca = pca.transform(np.array(features_test))
#    #clf.fit(features_train,labels_train)
#    clf.fit(features_train_pca,labels_train)
#    pred = clf.predict(features_test_pca)
#    f1_score_list.append(f1_score(labels_test,pred))        
#    #print classification_report(labels_test, pred)
#    #print confusion_matrix(labels_test, pred)
#    #print"SVC: %0.3f" % (clf.score(features_test_pca,labels_test))
## STRATIFIED TRIAL END ##


    
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## K-MEANS TRIAL BEGIN ##
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)
## K-MEANS TRIAL END ##

## INITIAL TRY BEGIN##
#features_train_pca = pca.fit_transform(np.array(features_train))
#features_test_pca = pca.transform(np.array(features_test))
#clf.fit(features_train,labels_train)
#clf2.fit(features_train,labels_train)
#clf3.fit(features_train,labels_train)
#clf4.fit(features_train,labels_train)
#print"K Means without PCA: %0.3f" % (clf.score(features_test,labels_test))
#print"Naive Bayes without PCA: %0.3f" % (clf2.score(features_test,labels_test))
#print"Decision Tree Classifier without PCA: %0.3f" % (clf3.score(features_test,labels_test))
#print"SVC without PCA: %0.3f" % (clf4.score(features_test,labels_test))
#clf.fit(features_train_pca,labels_train)
#clf2.fit(features_train_pca,labels_train)
#clf3.fit(features_train_pca,labels_train)
#clf4.fit(features_train_pca,labels_train)
#print"K Means with PCA: %0.3f" % (clf.score(features_test_pca,labels_test))
#print"Naive Bayes with PCA: %0.3f" % (clf2.score(features_test_pca,labels_test))
#print"Decision Tree Classifier with PCA: %0.3f" % (clf3.score(features_test_pca,labels_test))
#print"SVC with PCA: %0.3f" % (clf4.score(features_test_pca,labels_test))
## INITIAL TRY END ##


## SVC TRIAL BEGIN ##
#clf = SVC(kernel = 'rbf', C = 1e7, gamma = 1e-5) # low C: more support, tries to fit more (overfitting), high gamma: points influence region too low (i.e. islands of decision surfaces, overfitting ?)
#svc_obj = SVC(kernel = 'rbf', class_weight = 'balanced', C = 1e7, gamma = 1e-9)
#pca = decomposition.PCA()
#scaler = MinMaxScaler(feature_range = (-1, 1))
#clf = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('svc_obj', svc_obj)])
#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#print classification_report(labels_test, pred)
#print confusion_matrix(labels_test, pred)
#print"SVC: %0.3f" % (clf.score(features_test_pca,labels_test))
## SVC TRIAL END ##


# GRID SEARCH BEGIN ## 
#kernel = ['linear']
#svc_obj = SVC(class_weight = 'balanced')
#pca = decomposition.PCA()
#scaler = StandardScaler()
#clf = Pipeline(steps=[('scaler',scaler), ('pca', pca), ('svc_obj', svc_obj)])
#cv = StratifiedShuffleSplit(labels, n_iter=5, random_state=42)
#
#
#C = [1e-5,1, 1e3, 1e4, 1e5, 1e6]
#C = np.logspace(-2, 10, 13)
#gamma = [1e-9, 1e-8]
#gamma = np.logspace(-13, 0, 14)
#         
#estimator = GridSearchCV(clf,
#                         param_grid = dict(svc_obj__kernel = kernel,
#                              svc_obj__C = C, svc_obj__gamma = gamma), scoring = 'f1', cv = cv)
#estimator.fit(features, labels)
#print "SVC with PCA and grid search: %0.3f" % (estimator.score(features,labels))
##clf.score(features_test_pca,labels_test)
#print estimator.best_params_
#pred = estimator.predict(features)
#print classification_report(labels, pred)
#print confusion_matrix(labels, pred)
#clf = estimator.best_estimator_
## GRID SEARCH END ## 

#
#plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#            linestyle=':', label='n_components chosen')
#plt.legend(prop=dict(size=12))
#plt.show()

## OLD GRID SEARCH BEGIN ##
#features_train_pca = pca.fit_transform(np.array(features_train))
#features_test_pca = pca.transform(np.array(features_test))
#param_grid = {
#         'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
#          'gamma': [1e-10, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
#          }
## for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
##clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
##clf = clf.fit(features_train_pca, labels_train)
#clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv = 3)
#clf = clf.fit(features_train, labels_train)
#print "Best estimator found by grid search:"
#print clf.best_estimator_    
#print "SVC with PCA and grid search: %0.3f" % (clf.score(features_test,labels_test))
##clf.score(features_test_pca,labels_test)
#pred = clf.predict(features_test)
#print classification_report(labels_test, pred)
#print confusion_matrix(labels_test, pred)
## OLD GRID SEARCH END ##


#clf = clf3;
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)