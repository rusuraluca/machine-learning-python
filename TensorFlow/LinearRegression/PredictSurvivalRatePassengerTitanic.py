from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Load dataset

# .read_csv(): return to us a new pandas dataframe/table
# training data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
# testing data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
# we've decided to pop the "survived" column from our dataset and store it in a new variable
# this column simply tells us if the person survived our not
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# .head(): look at the data we'll use, show us the first 5 items in our dataframe
dftrain.head()

# .describe(): if we want a more statistical analysis of our data
dftrain.describe()

dftrain.shape

