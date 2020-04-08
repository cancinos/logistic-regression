# logistic-regression

This repository contains my first attempt to build a logistic regression using python. I'm neither an expert in python or machine learning. 
But I want give here my contribution to the community and for the ones who are trying to learn machine learning as I am.

Please feel free to contribute to this repository, or to comment anything about coding style, enhancements on algorithm, etc.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

#### Environment
- Python 3.7
- Conda

#### Libraries
- Pandas
- numpy
- math
- random


### Installing

Just git clone this repository in your local machine as

```bash
$ git clone https://github.com/cancinos/logistic-regression.git
```

## Deployment

I have two main functionalities on my code. In both you need to set three hyperparams, these are alpha (learning rate), the number of 
iterations per learning, and the threshold for the value of the activation function. 

1. **Setting a training set and an evaluation set** 
    - Inputs
    ```
    $ python logistic_regression.py
    $ 1. Where's your training set?
    $ --> winequality-red-training.csv
    $ 2. Are you going to eveluate it? y/n
    $ --> y
    $ 3. Where's your evaluation set?
    $ --> winequality-red-evaluating.csv
    $ 4. Give me your learning rate (alpha)
    $ --> 0.1
    $ 5. Give me your iterations number
    $ --> 1000
    $ 6. Give me your threshold
    $ --> 0.4
    ```
    - Ouputs
    ```
    $ Precision 0.152778  
    $ Recall    0.440000
    $ Elapsed time: 0.4029865264892578 secs
    ```
2. **Setting a training set and evaluate through 10-fold Cross-validation** 
    - Inputs
    ```
    $ python logistic_regression.py
    $ 1. Where's your training set?
    $ --> winequality-red-training.csv
    $ 2. Are you going to eveluate it? y/n
    $ --> n
    $ 3. Ok then, we are going to use 10-fold cross-validation for it!
    $ 4. Give me your learning rate (alpha)
    $ --> 0.1
    $ 5. Give me your iterations number
    $ --> 1000
    $ 6. Give me your threshold
    $ --> 0.4
    ```
    - Ouputs
    ```
        
    $ Precision 0.354365  
    $ Recall    0.441727
    $ Elapsed time: 6.06175684928894 secs
    ```
### Recommended hyperparams 

## Authors

* **Pablo Cancinos** - *Future Data Scientist* - [cancinos](https://github.com/cancinos)
