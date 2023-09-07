import sys
from numpy import *
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import random
from math import *

TARGET_ATTRIBUTE = 2 ## 0-23

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    is_first = True
    
    for line in fr.readlines():
        if is_first:
            is_first=False
            continue
        
        curLine = line.strip().split(',')[1:-1]
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def get_similarity(training_data,target_data):
    
    training_avg = mean(delete(training_data,TARGET_ATTRIBUTE,axis=1),axis=1).reshape((len(training_data),1))
    
    subtracted_training_data = training_data - repeat(training_avg,len(training_data[0]),axis=1)
    
    target_avg = mean(delete(target_data,TARGET_ATTRIBUTE))
    subtracted_target_data = target_data - repeat(target_avg,len(target_data))
    
    temp_subtracted_target_data = mat(delete(subtracted_target_data,TARGET_ATTRIBUTE))
    temp_subtracted_training_data = mat(delete(subtracted_training_data,TARGET_ATTRIBUTE,axis=1))
    temp_subtracted_target_data = transpose(temp_subtracted_target_data)
    numerater = dot(temp_subtracted_training_data,temp_subtracted_target_data)
    training_self_square = dot(temp_subtracted_training_data,transpose(temp_subtracted_training_data))
    target_self_square_and_squareroot = sqrt(dot(transpose(temp_subtracted_target_data),temp_subtracted_target_data))

    predicted_rating = target_avg
    weights = []
    for i in range(len(training_data)):
        weights.append(numerater[i]/(sqrt(training_self_square[i,i])*target_self_square_and_squareroot))
        
    weights = weights/sum(weights)
    for i in range(len(training_data)):
        predicted_rating+=(weights[i]*subtracted_training_data[i,TARGET_ATTRIBUTE])
    return predicted_rating[0][0]

def evaluate(training_data,testing_data):
    correct_count = 0
    c = 0
    for t in testing_data:
        #print(c)
        c+=1
        predicted_rating = get_similarity(training_data,t)
        if (predicted_rating >=3 and t[TARGET_ATTRIBUTE] >=3) or (predicted_rating < 3 and t[TARGET_ATTRIBUTE] <3):
            correct_count += 1
        
    return correct_count/len(testing_data)
if __name__ == '__main__':
    
    data_filename = 'google_review_ratings.csv'
    data = loadDataSet(data_filename)
    data_len = len(data)
    
    
    testing_data = [] 
    training_data = []
    for i in range(1000):
        t_i = random.choice(range(data_len-i))
        testing_data.append(data.pop(t_i))
    
    training_data = data #4456 
    
    # test 1 record

    #predicted_rating = get_similarity(training_data,testing_data[0])
    #print(predicted_rating)
    accuracies = []
    for idx in range(24):
        TARGET_ATTRIBUTE = idx
        accuracy = evaluate(training_data,testing_data)
        accuracies.append(accuracy)
        print(f'Attribute {TARGET_ATTRIBUTE}: {accuracy}')
    print(accuracies)
    
    # check the missing data of each attributes
    '''
    stats = [0]*24
    for r in data:
        for i,x in enumerate(r):
            
            if x == 0:
                stats[i]+=1
                
            
    print(stats)
    '''
            
    
    

