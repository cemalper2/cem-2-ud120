#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def visualize_financial_data(data_TBV,feature_no_1,feature_no_2,features):
    non_poi_data = data_TBV[data_TBV[:,0] == 0,:]
    poi_data = data_TBV[data_TBV[:,0] == 1,:]
    fig = plt.figure();
    plt.scatter(non_poi_data[:,feature_no_1],non_poi_data[:,feature_no_2],c='b')
    plt.scatter(poi_data[:,feature_no_1],poi_data[:,feature_no_2],c='r')
    plt.xlabel(features[feature_no_1])
    plt.ylabel(features[feature_no_2])
    fig.tight_layout()
    plt.savefig(features[feature_no_1] + '_' + features[feature_no_2] + '.png',dpi = 300) 
    plt.show()



### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred',
            'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
            'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person',
            'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
#data_dict.pop('LOWRY CHARLES P', 0)
#data_dict.pop('KAMINSKI WINCENTY J', 0)
#data_dict.pop('BHATNAGAR SANJAY', 0)
#data_dict.pop('DELAINEY DAVID W', 0)
#data_dict.pop('LAVORATO JOHN J', 0)








data = featureFormat(data_dict, features, sort_keys = True)
print(sorted(data_dict.keys())[data[:,11].argmax()])
print(sorted(data_dict.keys())[data[:,16].argmax()])
print(sorted(data_dict.keys())[data[:,17].argmax()])
print(sorted(data_dict.keys())[data[:,18].argmax()])
print(sorted(data_dict.keys())[data[:,5].argmax()])
print(sorted(data_dict.keys())[data[:,4].argmax()])
print(sorted(data_dict.keys())[data[:,3].argmax()])

print(data_dict[sorted(data_dict.keys())[data[:,11].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,16].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,17].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,18].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,5].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,4].argmax()]]['poi'])
print(data_dict[sorted(data_dict.keys())[data[:,3].argmax()]]['poi'])


### your code below
for feat_1 in range(0,len(features)):
    visualize_financial_data(data,0,feat_1,features)

