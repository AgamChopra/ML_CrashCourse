'''
ML Crash Course
Author: Agamdeep S. Chopra

Assignment 1 - Image classification with MNIST 1-5

Install the following packages as needed.
'''
#!pip install PIL
#!pip install gzip
#!pip install matplotlib
#!pip install numpy
#!pip install sklearn
#!pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#%%
PATH = "E:/ML/mnist" # <- Change path accordingly.
import sys
sys.path.append(PATH)
import dataset as db
db.PATH = PATH
import models
import torch
#%%
# Load the Datasets
tr,ts,vl = db.dataset(True)
# Optional - Explore the dataset
print(len(tr), len(tr[0]), len(tr[1]))
print(tr[0].shape, tr[1].shape)
db.plot_image(tr[0], 47)
# Q1 Extract Training Images and Labels
train_x = 
train_y = 
# Q2 Extract Validation Images and Labels
val_x = 
val_y = 
# Q3 Extract Test Images and Labels
test_x = 
test_y = 
#%%
###LogisticRegression###
# Q4 Call the predefined Logistic Regression model
model = 
# Q5 Fit/Train the model on the training dataset

# Print Accuracy Statistics
print('LogisticRegression')
print('Train Accuracy:',models.accuracy(train_y,model.predict(train_x)))
print('Validation Accuracy:',models.accuracy(val_y,model.predict(val_x)))
print('Test(custom dataset) Accuracy:',models.accuracy(test_y,model.predict(test_x)))
#Expected Accuracy:
#    LogisticRegression
#    Test Accuracy: tensor(0.9339)
#    Evaluation Accuracy: tensor(0.9255)
#    Test(custom dataset) Accuracy: tensor(0.3600) 
#%%