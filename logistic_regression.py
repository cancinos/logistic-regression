"""
    Import libraries
""" 
import sys
from logreg_functions import *
from decimal import Decimal

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
alpha = Decimal(input('--> '))

print("4. Give me your iterations number")
numIterations = int(input('--> ')) 

print("4. Give me your threshold")
threshold = Decimal(input('--> ')) 

"""
Datasets:
    training_ds -> We load this dataset in order to train our model
    ? evaluation_ds -> We load this dataset in order to evaluate our model
"""

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
"""
model_outcome = logistic_reggression(not eval_type, alpha, threshold, numIterations, training_ds, evaluation_ds)





