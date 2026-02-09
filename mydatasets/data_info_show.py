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
x = train_data[0]
y = train_data[1]
for i in range(50):
    print(f"x[{i}]: {x[i]}") # #1 sample session
    print(f"y[{i}]: {y[i]}") # #1 sample target

