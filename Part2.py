"""
Created on Tue May  9 12:18:13 2023

@author: Kaan Gursoy
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score


digit_data = np.loadtxt("digits.txt")
label_data = np.loadtxt("labels.txt")

xtrain, xtest, ytrain, ytest = train_test_split(digit_data, label_data, stratify=label_data)

# Isomap
training_error_list = []
testing_error_list = []
iso_dimension = np.linspace(1, 200, 20, dtype=int)


for k in iso_dimension:
    isomap = Isomap(n_components=k)
    isomap.fit(xtrain)   
    xtrain_new = isomap.transform(xtrain)
    xtest_new = isomap.transform(xtest)
    gausclassifier = GaussianNB()
    gausclassifier.fit(xtrain_new, ytrain)
    trainerror = 1 - gausclassifier.score(xtrain_new, ytrain)
    testerror = 1 - gausclassifier.score(xtest_new, ytest)
    training_error_list.append(trainerror)
    testing_error_list.append(testerror)
    print(f"Dimension: {k}, Training Error: {trainerror:.4f}, Test Error: {testerror:.4f}")


plt.figure()
plt.plot(iso_dimension, training_error_list, label="Training Error",marker="o")
plt.plot(iso_dimension, testing_error_list, label="Test Error",marker="o")
plt.xlabel("iso_dimension")
plt.ylabel("Classification Error")
plt.title("Classification Error vs. iso_dimension (Isomap)")
plt.legend()
plt.show()

