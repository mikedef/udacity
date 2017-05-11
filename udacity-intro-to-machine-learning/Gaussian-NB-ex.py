#!/usr/bin/env python 
import numpy as np
# Features
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Lables
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB() # Creates a classifier 
# .fit() will fit gaussian naive bayes according to X,y
clf.fit(X, Y) # Give the classifier the training data (Features,Lables)

print(clf.predict([[-0.8, -1]])) # Predict the where the data belongs. 

# Create a second classifier
clf_pf = GaussianNB()
# partial_fit() will incrementally fit on a batch of samples
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[-0.8, -1]]))
