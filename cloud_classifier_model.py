
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:42:46 2018

@author: jkcm
"""

def load_training_data():
    raise NotImplementedError
    data, labels = [], []
    return data, labels

# def split_training_data(data, labels):
#     raise NotImplementedError
#     train, test, validate = []
#     return train, test, validate




def initialize_model(model_params):
    raise NotImplementedError
    return model

def train_model(model, train_data, train_labels, train_params):
    raise NotImplementedError
    return model

def validate_model(model, validate_data, validate_label):
    raise NotImplementedError
