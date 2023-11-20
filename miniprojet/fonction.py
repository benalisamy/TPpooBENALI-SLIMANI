#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ce code est une implementation de l'algorithme de classification K-nn 
#sur une petite base de données . en utilisant la distance eucledienne
#travail fait par BENALI SAMY //// SLIMANI IKRAM 


# In[2]:


import csv
import random
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import operator
import pandas as pd

# Fonction pour charger notre data base  à  partir d'un fichier CSV et la diviser en ensembles d'entraînement et de test
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# Initialisation des ensembles d'entraînement et de test
trainingSet = []
testSet = []

# Chargement des données du fichier 'iris.data.csv'
loadDataset('iris.data.csv', 0.66, trainingSet, testSet)
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))


# In[3]:


# Fonction pour le calcul de la distance euclidienne
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# Exemple de calcul de distance euclidienne entre deux points
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance: ' + repr(distance))


# In[1]:


# Fonction pour le calcul de la distance de  Manhattan
def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance


# Test the euclideanDistance function
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = manhattanDistance(data1, data2, 3)
print('Distance: ' + repr(distance))


# In[4]:


# Fonction pour trouver les k voisins les plus proches d'une instance de test dans l'ensemble d'entraînement
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Exemple d'utilisation de la fonction getNeighbors
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(f"test du plus proche voisin est : {neighbors}")


# In[ ]:


# Fonction pour trouver les k voisins les plus proches d'une instance de test dans l'ensemble d'entraînement en utilisant 
#la distance de manhattan
def getNeighborsM(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = manhattanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Exemple d'utilisation de la fonction getNeighbors
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(f"test du plus proche voisin est : {neighbors}")


# In[5]:


# Fonction pour obtenir la classe avec le plus grand nombre de votes parmi les voisins les plus proches
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Exemple d'utilisation de la fonction getResponse
neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
response = getResponse(neighbors)
print(f"teste de la réponse : {response}")


# In[6]:


# Fonction pour calculer la précision des prédictions obtenus 
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

# Exemple d'utilisation de la fonction getAccuracy
test = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predi = ['a', 'a', 'a']
testaccuracy = getAccuracy(test, predi)
print(f"teste de la fonction précision : {testaccuracy}")


# In[ ]:


# Fonction principale
def main():
    predictions = []
    actual_labels = []
    k = 3

    # Faire des prédictions pour chaque instance de l'ensemble de test
    for x in range(len(testSet)):
         neighbors = getNeighbors(trainingSet, testSet[x], k)
         result = getResponse(neighbors)
         predictions.append(result)
         actual_labels.append(testSet[x][-1])  # Ajouter l'étiquette actual
         print('> Predicted=' + repr(result) + ', Actual=' + repr(testSet[x][-1]))

    # Ajout des prédictions à  notre ensemble de test
    testSetWithPredictions = [testSet[i] + [predictions[i]] for i in range(len(testSet))]

    # Calculer et afficher la précision des prédictions
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    # Calculer et afficher la matrice de confusion
    cm = confusion_matrix(actual_labels, predictions)
    print("\nConfusion Matrix:")
    print(np.array2string(cm, separator=', '))

    # Afficher une  matrice de confusion
    classes = ['setosa', 'versicolor', 'virginica']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Comparer les comptes de classes réelles et prédites avec un diagramme en barres
    unique_labels = set(actual_labels + predictions)
    unique_labels = sorted(list(unique_labels))

    actual_counts = [actual_labels.count(label) for label in unique_labels]
    predicted_counts = [predictions.count(label) for label in unique_labels]

    bar_width = 0.35
    index = np.arange(len(unique_labels))

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, actual_counts, bar_width, label='Actual')
    bar2 = ax.bar(index + bar_width, predicted_counts, bar_width, label='Predicted')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of Actual and Predicted Classes')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(unique_labels)
    ax.legend()

    plt.show()


# In[7]:


# Appeler la fonction principale
#main()


# In[ ]:




