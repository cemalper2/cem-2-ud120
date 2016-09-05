#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import math
    error = [(ages[array_ind][0], net_worths[array_ind][0],  math.fabs(prediction - net_worths[array_ind])) for (array_ind, prediction) in enumerate(predictions)]
    cleaned_data = sorted(error, key = lambda tup: tup[2])
    print cleaned_data[0:9]
    print (math.floor(len(cleaned_data)*0.1))
    ### your code goes here
    cleaned_data = cleaned_data[:(len(cleaned_data) - int(math.floor(len(cleaned_data)*0.1)))]
    
    
    return cleaned_data

