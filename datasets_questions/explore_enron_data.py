#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

poi_count = 0
for key in enron_data:
    poi_count+=1 if enron_data[key]['poi'] else 0
    
with_salary = 0
for key in enron_data:
    with_salary+=1 if type(enron_data[key]['salary']) is int else 0



with_email = 0
for key in enron_data:
    with_email+=0 if enron_data[key]['email_address'] == 'NaN' else 1
    
nan_total_payments = 0
poi_with_nan_payments = 0
for key in enron_data:
    nan_total_payments+=1 if enron_data[key]['total_payments'] == 'NaN' else 0
    poi_with_nan_payments+=1 if (enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi']) else 0