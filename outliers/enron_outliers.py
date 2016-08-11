#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)


### your code below
max_bonus = data[0][1]
max_ind = 0
for (ind,point) in enumerate(data):
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary,bonus)
    if point[1]>max_bonus:
        max_bonus = point[1]
        max_ind = ind
matplotlib.pyplot.xlabel('salary')
matplotlib.pyplot.ylabel('bonus')
matplotlib.pyplot.show()

