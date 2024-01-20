Lenet5
======

This code will train and save a Lenet5 model.\
It will explore the following architectures:
1. Regular Lenet5
2. Batch Normalization
3. Weight Decay
4. Dropout

It will train on FashionMNIST dataset

Instructions:
-----------

1. The main code to run is under the script **main.py**\
2. The train & test data are under the directory: data/fashion

TODOs:
-----------
1. Split training data into Train & Validations sets
2. Use the validation set instead of the test set as "validation", evaluate the accuracy on the test set only once 
   at the end
3. Implement hyperparamers search & optimization functions
4. Choose optimal hyper parameters based on (1):
   * learning rate
   * weight decay
   * dropout ratio
   * #epochs
   * batch size
   * optimizer
