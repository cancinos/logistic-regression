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

> "_If correctly identifying positives is important for us, then we should choose a model with higher Sensitivity. 
However, if correctly identifying negatives is more important, then we should choose specificity as the measurement metric._" Parul Pandey

In order to deeply understand this I highly recommend to read this article first [Simplifying the ROC and AUC metrics](https://towardsdatascience.com/understanding-the-roc-and-auc-curves-a05b68550b69). It made me realize what I have to do in order to achieve the hyperparams that optimize my model. One of the things that I realize was that the threshold value is one of the most important hyperparams that we have, this is because that I can control the output we have on our classification. If it is too low, we can increase false positive and on other side if it is too high, we can increase our false negatives. **In my case I decided that I'll bet for my specificity about my dataset**, and that's how I decided the following.



What I did was to fixed my learning rate at 0.00001, my iteration number at 1,000 and I iterate over my threshold value giving it increases of 0.01, and this were the results:

<p align="center">
  <img width="581" height="324" src="https://github.com/cancinos/logistic-regression/blob/qa_branch/graph_precision_recall.PNG">
</p>

**I highly recommend using a threshold of 0.2 in order to increse our specificity.**

## Authors

* **Pablo Cancinos** - *Future Data Scientist* - [cancinos](https://github.com/cancinos)
