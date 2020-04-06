"""
    Import libraries
""" 
import sys
from logreg_functions import *
import pandas as pd
import time

"""
    Consoles outputs and inputs:
        1. Say Hi and ask for training set, evaluation set, alpha, numIterations and threshold
"""
print("Welcome to your logistic regression engine! Please give me the following variables: ")

print("1. Where's your training set?")
training_set_path = input('--> ')

print("2. Are you going to eveluate it? y/n")
evaluate_decision = input('--> ')

if evaluate_decision == 'y' :
    print("3. Where's your evaluation set?")
    evaluation_set_path = input('--> ')
    eval_type = 1
else :
    print("3. Ok then, we are going to use 10-fold cross-validation for it!")
    eval_type = 0

print("4. Give me your learning rate (alpha)")
alpha = float(input('--> '))

print("4. Give me your iterations number")
numIterations = int(input('--> ')) 

print("4. Give me your threshold")
threshold = float(input('--> ')) 

"""
Datasets:
    training_ds -> We load this dataset in order to train our model
    ? evaluation_ds -> We load this dataset in order to evaluate our model
"""
start = time.time()
training_ds = pd.read_csv(training_set_path)

if eval_type :
    evaluation_ds = pd.read_csv(evaluation_set_path)

"""
Feature engineering:
    Modifying our datasets in order to run the logistic regression model
"""
training_ds = feature_engineering(training_ds)

if eval_type :
    evaluation_ds = feature_engineering(evaluation_ds)
else :
    evaluation_ds = None

"""
Two types of evaluation:
    1. Run logistic regression, then validate with evaluation set and present precision and recall values.
    2. Run logistic regression, then run 10-fold cross-validation with training set and present precision and
       recall values.
        2.1 '1' = training and '0' for evaluating
"""
if eval_type :
    model_outcome = logistic_reggression(alpha, threshold, numIterations, training_ds, evaluation_ds)
    print(model_outcome)
else : 
    factor = len(training_ds.index) / 10
    training_ds = training_ds.reindex(np.random.permutation(training_ds.index))
    training_ds = training_ds.reset_index(drop = True)
    outcomes = pd.DataFrame(columns=["Precision", "Recall"])

    for currentFold in range(0, 10):
        print ("Your current progress: ", currentFold/10*100,"% complete ", end='\r', flush=True)

        cross_validation_folding(currentFold, factor, training_ds)
        dataset = training_ds.copy()
        training_set = dataset[dataset.training == 1].iloc[: , 0 : 12].reset_index(drop = True)
        evaluation_set = dataset[dataset.training == 0].iloc[: , 0 : 12].reset_index(drop = True)

        model_outcome = logistic_reggression(alpha, threshold, numIterations, training_set, evaluation_set)
        outcomes = outcomes.append(model_outcome) 
        
    print ("")
    print(outcomes)
    print(outcomes.mean())
        
end = time.time()
print(end - start)




