---
layout: post
title: "URL Classifier"
---
### Contents
- [Summary](#1)
- [GitHub Repository](#2)
- [Project Details](#3)
    - [Initialization](#3.1)
        - [Environment Setup](#3.1.1)
        - [Install prerequisite libraries](#3.1.2)
        - [Import Packages to Use](#3.1.3)
        - [Global Variables](#3.1.4)
    - [Data Processing](#3.2)
        - [Load Data](#3.2.1)
        - [Split Data](#3.2.2)
        - [Preprocess Data](#3.2.3)
        - [TensorFlow Dataset Pipeline](#3.2.4)
    - [Modeling](#3.3)
        - [Model Construction](#3.3.1)
        - [Model Training](#3.3.2)
    - [Result](#3.4)
        - [Loss Rate and Accuracy](#3.4.1)
        - [Precision and Recall](#3.4.2)

------

<h3 id="1">Summary</h3>
This project is about constructing a URL classification model using the TensorFlow library. The model is trained to classify URLs into three categories: 'left', 'central', and 'right'. The project is organized as a Jupyter notebook (Classifier_Demo.ipynb) and a utility script (utils.py).

------

<h3 id="2">GitHub Repository</h3>
<!-- TODO: Add your link -->
Check the source codes out on my [GitHub repo](https://github.com/Nonnature/URL_Classifier).

------

<h3 id="3">Project Details</h3>

- <h4 id="3.1">Initialization</h4>

Firstly, the project starts with setting up the environment and importing necessary packages, including TensorFlow, pandas, numpy, and others.
    
1. <h6 id="3.1.1">Environment Setup</h6>

The environment setup includes mounting Google Drive and specifying the directory for data and model resources.

```python
# URL Classifier
""" Prepare Notebook for Google Colab """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Specify location
module_dir = (
    "/content/drive/My Drive/Demo"
)

# Add material directory
import sys
sys.path.append(module_dir)
```

2. <h6 id="3.1.2">Install prerequisite libraries</h6>

```python
""" Install prerequisite libraries """
# Some of the codes in this notebook may return error
# with different versions of the packages listed below.
# Please install the packages with the versions specified below.
!pip3 install wordninja==2.0.0             # for splitting joined words
!pip3 install scikit-learn==0.23.2         # for one-hot encoding
!pip3 install lime==0.2.0                  # for explaining model predictions

# Uncomment to install the libraries if necessary
#!pip3 install tensorflow==2.0.0           # if you use TensorFlow w/out GPU
!pip3 install tensorflow-gpu==2.0.0       # if you use TensorFlow w/ GPU
```
3. <h6 id="3.1.3">Import Packages to Use</h6>

```python
""" Import Packages to Use """

import os
import random
import numpy as np
import pandas as pd
import pickle

# For splitting joined words
import wordninja

# Import TensorFlow
import tensorflow as tf
# Text processing methods with TF
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Check if any GPU is detected
print("Is GPU available: ", tf.test.is_gpu_available())
print("GPU(s) found: ")
print(tf.config.experimental.list_physical_devices('GPU'))

# For visualization
import matplotlib.pyplot as plt
%matplotlib inline

# For explaining URL classification predictions
from lime.lime_text import LimeTextExplainer

# We provide a function to prepare data as inputs to our classificaiton model.
# About how the data are processed, please refer to: text_processing_demo.ipynb.
from utils import preprocess_data
# Also, a function is prepared to takes URL string as input and returns the
# predicted category.
from utils import predict_url
```

4. <h6 id="3.1.4">Global Variables</h6>

```python
""" Global Variables """

# Set random seed
RAND_SEED = 9999
random.seed(RAND_SEED)
# Random seed for TensorFlow
tf.random.set_seed(RAND_SEED)

##### Data #####
# For demo purpose, we only use data of a subset of categories in this notebook
CATEGORIES_TO_USE = ['left','central','right']
N_CLASSES = len(CATEGORIES_TO_USE) # Number of classes to predict
# Size of training dataset
TRAIN_SIZE = 0.8

##### Text Processing #####
# Vocabulary size for tokenization
VOCAB_SIZE = 10000
# Maximum length of token sequence
MAX_LEN = 20

##### Classification Model #####
# Size of data batches
BATCH_SIZE = 40000
# Dimension of Word Embedding layer
EMBEDDING_DIM = 512
# Learning rate of optimizer
LR = 1e-3
# Number of epochs to train model
N_EPOCH = 100
```

- <h4 id="3.2">Data Processing</h4>

1. <h6 id="3.2.1">Load Data</h6>

```python
""" Load Data """

# Specify root data directory
data_dir = os.path.join(module_dir, "data/")

# Path to data file
data_path = os.path.join(data_dir, "URL.csv")

# Load data into pandas DataFrame
data_df = pd.read_csv(data_path, header=None)
print("Example Data:")
print(data_df.head())
```

2. <h6 id="3.2.2">Split Data</h6>

```python
""" Split Data into Training & Test Sets """

# First, shuffle the data, by shuffling indices.
idx_all = list(range(len(data_df)))
random.shuffle(idx_all)

# Find out where to split training/test set
m = int(len(data_df) * TRAIN_SIZE)

# Split indices
idx_train, idx_test = idx_all[:m], idx_all[m:]
# Split data
X_train, y_train = data_df.iloc[idx_train, 1].values, data_df.iloc[idx_train, 2].values
X_test, y_test = data_df.iloc[idx_test, 1].values, data_df.iloc[idx_test, 2].values

# Show dataset stats
print("Training set label distriubtion: ", np.unique(y_train, return_counts=True))
print("Test set label distriubtion: ", np.unique(y_test, return_counts=True))
```

3. <h6 id="3.2.3">Preprocess Data</h6>

```python
""" Preprocess Data """

print("===== Before Preprocessing =====")
print("----- Training Data -----")
print("URL: ", X_train[0])
print("Category: ", y_train[0])
print("----- Test Data -----")
print("URL: ", X_test[0])
print("Category: ", y_test[0])
print("================================")


# First, we process the training data. While processing the training data, a
# tokenizer and an encoder are constructed based on the training inputs and
# labels, resp. The function preprocess_data(...) returns the tokenizer and
# encoder used.
X_train, y_train, tokenizer, encoder = preprocess_data(
    X_train, y_train, return_processors=True,
)

# Based on the tokenizer and encoder constructed from training data, we process
# the test data here too.
X_test, y_test = preprocess_data(
    X_test, y_test,
    tokenizer=tokenizer, encoder=encoder,
)

print("===== After Preprocessing =====")
print("----- Training Data -----")
print("Input: ", X_train[0])
print("Label: ", y_train[0])
print("----- Test Data -----")
print("Input: ", X_test[0])
print("Label: ", y_test[0])
print("================================")
```

4. <h6 id="3.2.4">TensorFlow Dataset Pipeline</h6>

```python
""" TensorFlow Dataset Pipeline """

# A TF dataset pipeline is prepared here for feeding input batches to train the
# model later.

# Training set data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))    # Input: tokenized urls, target labels
train_ds = train_ds.shuffle(buffer_size=len(X_train))                # Shuffle training data
train_ds = train_ds.batch(batch_size=BATCH_SIZE)                     # Split shuffled training data into batches
# Test set data pipeline
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(batch_size=BATCH_SIZE)
```

- <h4 id="3.3">Modeling</h4>

1. <h6 id="3.3.1">Model Construction</h6>

```python
# Construct a neural network
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(LR),
              metrics=['accuracy'])

# Show model structure
print(model.summary())
```

2. <h6 id="3.3.2">Model Training</h6>

```python
""" Train Model """

history = model.fit(train_ds, epochs=N_EPOCH,
                    validation_data=test_ds, verbose=1)
```

- <h4 id="3.4">Result</h4>

1. <h6 id="3.4.1">Loss Rate and Accuracy</h6>

```python
""" Visualize Training Results """

##### Loss #####
# Get training results
history_dict = history.history
train_acc = history_dict['loss']
test_acc = history_dict['val_loss']

# Plot training results
plt.plot(train_acc, label='Train Loss')
plt.plot(test_acc, label='Test Loss')

# Show plot
plt.title('Model Training Results (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="lower right")
plt.grid()
plt.show()

##### Accuracy #####
# Get training results
history_dict = history.history
train_acc = history_dict['accuracy']
test_acc = history_dict['val_accuracy']

# Plot training results
plt.plot(train_acc, label='Train Acc.')
plt.plot(test_acc, label='Test Acc.')

# Show plot
plt.title('Model Training Results (Acc.)')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(loc="lower right")
plt.grid()
plt.show()
```

<div align=center><img src="{{ site.url }}/assets/url_lossacc.png" height="60%" width="60%" style="margin: 5%" style="margin: 5%"></div>

2. <h6 id="3.4.2">Precision and Recall</h6>

```python
""" Precision & Recall """

# Sklearn provides the classificatiaon_report(...) function which makes
# evaluation of classification model easy.
from sklearn.metrics import  classification_report

print(classification_report(
    np.argmax(y_test, axis=1),  # ground-truths
    np.argmax(model.predict(X_test), axis=1), # predictions
    target_names=CATEGORIES_TO_USE)
)
```

<div align=center><img src="{{ site.url }}/assets/url_precall.png" height="60%" width="60%" style="margin: 5%" style="margin: 5%"></div>