"""
    Import libraries
""" 
import numpy as np
import pandas as pd
import math as m


def feature_engineering(dataset) :

    """
        Update our `label` values to 1s and 0s
    """

    dataset['quality'] = np.where(
        dataset['quality'] == "Bad",
        0,
        np.where(
            dataset['quality'] == "Good",
            1,
            -1
        )
    )

    return dataset

def logistic_reggression(cross_validation, alpha, threshold, numIterations, training_set, evaluation_set) :

    """
        Initialize weights
    """
    weights = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    #if not cross_validation :

    for currentIteration in range(1, numIterations):
        weights = math_engine(currentIteration, training_set, weights, threshold, alpha)
        print(weights)

    return 0


def math_engine(iteration, training_set, weights, threshold, alpha) :

    """
        3.1.1. First we calculate general values of the model, which means 
        calculating y, y^ and its threshold
    """    
    alpha = float(alpha)
    threshold = float(threshold)

    fixed_acidity = weights.iloc[1] * training_set['fixed acidity']
    volatile_acidity = weights.iloc[2] * training_set['volatile acidity']
    citric_acid = weights.iloc[3] * training_set['citric acid']
    residual_sugar = weights.iloc[4] * training_set['residual sugar']
    chlorides = weights.iloc[5] * training_set['chlorides']
    free_sulfur_dioxide = weights.iloc[6] * training_set['free sulfur dioxide']
    total_sulfur_dioxide = weights.iloc[7] * training_set['total sulfur dioxide']
    density = weights.iloc[8] * training_set['density']
    pH = weights.iloc[9] * training_set['pH']
    sulphates = weights.iloc[10] * training_set['sulphates']
    alcohol = weights.iloc[11] * training_set['alcohol']

    y = weights.iloc[0] + fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol

    y_hat = pd.DataFrame(y).apply(lambda row : sigmod(row[0]), axis = 1)

    y_threshold = pd.DataFrame(y_hat).apply(lambda row : threshold_function(row[0], threshold), axis = 1)

    """
        3.1.2. Next we'll calculate loss function and its weights prime value for every row
    """    

    #loss = pd.DataFrame(y_hat).apply(lambda row : loss_function(row[0], training_set.iloc[row.name]), axis = 1)
    weights_prime = pd.DataFrame(y_hat).apply(lambda row : prime_function(row[0], training_set.iloc[row.name]), axis = 1)
    
    """
        3.1.3. Then we calculate mean values for the loss function and the prime weights 
    """    
    delta_w = pd.DataFrame(weights_prime).mean()

    print("------------------------------------------------------------")

    return pd.Series([weights[0] - alpha * delta_w[0], 
        weights[1] - alpha * delta_w[1], 
        weights[2] - alpha * delta_w[2],
        weights[3] - alpha * delta_w[3],
        weights[4] - alpha * delta_w[4],
        weights[5] - alpha * delta_w[5],
        weights[6] - alpha * delta_w[6],
        weights[7] - alpha * delta_w[7],
        weights[8] - alpha * delta_w[8],
        weights[9] - alpha * delta_w[9],
        weights[10] - alpha * delta_w[10],
        weights[11] - alpha * delta_w[11],
    ])
    

def sigmod(y) :

    if y < 0 :
        return 1 - 1 / (1 + m.exp(y))

    return 1 / (1 + m.exp(-y))

def threshold_function(y_hat, threshold) :
    if y_hat >= threshold :
        return 1
    else :
        return 0

def loss_function(y_hat, real_value) :
    return -real_value['quality'] * m.log(y_hat) - (1 - real_value['quality']) * m.log(1-y_hat)

def prime_function(y_hat, training_value) :

    """
        Calculating W0'
    """    
    w_intercept = (y_hat - training_value['quality']) * 1
    
    """
        Calculating Wx'
    """    
    W_x = [w_intercept, 
    (y_hat - training_value['quality']) * training_value['fixed acidity'],
    (y_hat - training_value['quality']) * training_value['volatile acidity'], 
    (y_hat - training_value['quality']) * training_value['citric acid'], 
    (y_hat - training_value['quality']) * training_value['residual sugar'], 
    (y_hat - training_value['quality']) * training_value['chlorides'], 
    (y_hat - training_value['quality']) * training_value['free sulfur dioxide'], 
    (y_hat - training_value['quality']) * training_value['total sulfur dioxide'], 
    (y_hat - training_value['quality']) * training_value['density'], 
    (y_hat - training_value['quality']) * training_value['pH'], 
    (y_hat - training_value['quality']) * training_value['sulphates'], 
    (y_hat - training_value['quality']) * training_value['alcohol']
    ]

    W_x_row = pd.Series(W_x)

    return W_x_row