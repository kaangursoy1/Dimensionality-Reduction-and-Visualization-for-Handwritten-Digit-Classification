"""
Created on Tue May  9 12:18:13 2023

@author: Kaan Gursoy
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

digit_data = np.loadtxt("digits.txt")
label_data = np.loadtxt("labels.txt")

tsne = TSNE(n_components=2)

data_2d = tsne.fit_transform(digit_data)

plt.figure()
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=label_data, cmap='viridis', s=20)
plt.colorbar(scatter)
plt.title("t-SNE Visualization")
plt.ylabel("Dimension 2")
plt.xlabel("Dimension 1")
plt.show()
