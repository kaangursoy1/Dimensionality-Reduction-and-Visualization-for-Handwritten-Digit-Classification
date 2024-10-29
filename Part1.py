
"""
Created on Tue May  9 12:18:13 2023

@author: Kaan Gursoy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

n_components = 63

digit_data = np.loadtxt('digits.txt')
label_data = np.loadtxt('labels.txt')

xtrain, xtest, ytrain, ytest = train_test_split(digit_data, label_data, stratify=label_data, test_size=0.5)

#normalizing data
'''
mean = np.mean(digit_data, axis=0)
std = np.std(digit_data, axis=0)
digit_data = (digit_data - mean) / std

mean = np.mean(label_data, axis=0)
std = np.std(label_data, axis=0)
label_data = (label_data - mean) / std
'''
pca = PCA()
scaler = StandardScaler(with_std=False)
xtrain_normalized = scaler.fit_transform(xtrain)
xtest_normalized = scaler.transform(xtest)

pca.fit(xtrain_normalized)

plt.plot(pca.explained_variance_)
plt.xlabel('Number of Components')
plt.ylabel('Eigenvalues')
plt.show()

# optimal number of components based on the plot (it flattens at 100)

#20x20
sample_mean = scaler.mean_.reshape(20, 20)
plt.imshow(sample_mean, cmap='gray')
plt.show()

for i in range(n_components):
    plt.imshow(pca.components_[i].reshape(20, 20), cmap='gray')
    plt.show()

training_error_list = []
testing_error_list = []

subspace_dimensions = np.linspace(1, 200, 20, dtype=int)

for n_components in subspace_dimensions:

    pca = PCA(n_components=n_components)
    xtrain_pca = pca.fit_transform(xtrain_normalized)
    xtest_pca = pca.transform(xtest_normalized)
    classifier = GaussianNB()
    classifier.fit(xtrain_pca, ytrain)

    ytrain_pred = classifier.predict(xtrain_pca)
    trainerror = 1 - accuracy_score(ytrain, ytrain_pred)
    training_error_list.append(trainerror)

    ytest_pred = classifier.predict(xtest_pca)
    testerror = 1 - accuracy_score(ytest, ytest_pred)
    testing_error_list.append(testerror)
    print(f"Dimension: {n_components}, Training Error: {trainerror:.4f}, Test Error: {testerror:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(subspace_dimensions, training_error_list, label="Training Error", marker="o")
plt.plot(subspace_dimensions, testing_error_list, label="Test Error", marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Error")
plt.legend()
plt.title("Errors vs. Number Of Components ")
plt.grid()
plt.show()

mean_image = np.mean(xtrain, axis=0).reshape(20, 20)
plt.imshow(mean_image, cmap="gray")
plt.title("Sample Mean Image")
plt.show()

pca = PCA(n_components=100)
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(20, 20), cmap="gray")
    ax.set_title(f"PC {i+1}")

plt.show()