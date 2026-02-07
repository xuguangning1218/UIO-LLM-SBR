import argparse
import os
import pickle
import numpy as np

datasets = ["diginetica", "Tmall", "RetailRocket_DSAN", "Nowplaying"]

train_data = pickle.load(
        open("/data/UIO-LLM-SBR/datasets/" + datasets[0] + "/train.txt", "rb"))

print("train_data: ", len(train_data))
print("train_data[0]: ", len(train_data[0])) # session 
print("train_data[1]: ", len(train_data[1])) # target
print("train_data[0][1]: ", train_data[0][1]) # #1 sample session
print("train_data[1][1]: ", train_data[1][1]) # #1 sample target

