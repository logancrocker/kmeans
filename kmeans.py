from copy import deepcopy
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import random
import warnings

#warnings.filterwarnings("ignore", category=RuntimeWarning) 

def dist(a, b):
    return np.linalg.norm(a - b)

data = pd.read_csv('iris.data', header=None, usecols=range(0,4))
#print(data.shape)
data.head()

#setting up input parameters
x_input = data.iloc[:, [0, 1, 2, 3]].values
k = 3
#print(x.shape)
#picking k unique random points as initial centroids
init_centroids = x_input[np.random.choice(x_input.shape[0], k, replace=False), :]
#print(init_centroids)

def k_means_cs171(x_input, k, init_centroids):
    #print(x_input.shape)
    curr_centroids = init_centroids
    #print(curr_centroids)
    old_centroids = []
    while np.array_equal(old_centroids, curr_centroids) != True:
        classifications = []
        #calculate the distance b/w given point and each centroid
        #assign it to closest centroid
        #print(x_input[0])
        #print(init_centroids)
        for point in x_input:
            min_distance = float(1000)
            for c in range(k):
                distance = dist(point, curr_centroids[c])
                #print(distance)
                if distance < min_distance:
                    min_distance  = distance
                    classification = c
            classifications.append((point, classification))
        #print(classifications)
        #now average the points for each centroid and make that the new centroid
        #first store the old list of centroids
        old_centroids = curr_centroids
        new_centroids = np.zeros((k, 4))
        for c in range(k):
            cen_list = [ item[0] for item in classifications if item[1] == c]
            #print(np.mean(cen_list, axis=0))
            #print(len(cen_list))
            #this if statement prevents a very annoying warning with numpy mean function
            if len(cen_list) != 0:
                new_centroids[c] = np.nanmean(cen_list, axis=0)
            else:
                new_centroids[c] = np.zeros((1, 4))
        curr_centroids = new_centroids
        #print(old_centroids)
        #print(curr_centroids)
    cluster_assignments = classifications
    cluster_centroids = curr_centroids
    return cluster_assignments, cluster_centroids
        

#part 1
(cluster_assignments, cluster_centroids) = k_means_cs171(x_input, k, init_centroids)
sum = 0
for c in range(k):
    cen_list = [ item[0] for item in cluster_assignments if item[1] == c]
    for val in cen_list:
        sum += pow(dist(val, cluster_centroids[c]), 2)
print("SSE: " + str(sum))

#part 2
def sensitivity_analysis(max_iter):
    y_values = []
    errors = []
    for i in range(10):
        k = i + 1
        sse_values = []
        for i in range(max_iter):
            init_centroids = x_input[np.random.choice(x_input.shape[0], k, replace=False), :]
            (cluster_assignments, cluster_centroids) = k_means_cs171(x_input, k, init_centroids)
            sum = 0
            for c in range(k):
                cen_list = [ item[0] for item in cluster_assignments if item[1] == c]
                for val in cen_list:
                    sum += pow(dist(val, cluster_centroids[c]), 2)
            #print("SSE: " + str(sum))
            sse_values.append(sum)
        print(sse_values)
        meanval = np.nanmean(sse_values)
        print(meanval)
        y_values.append(np.nanmean(sse_values))
        print(np.std(sse_values))
        errors.append(np.std(sse_values))
    return y_values, errors

max_iter = 1
(y_values, errors) = sensitivity_analysis(max_iter)

print(y_values)
print(errors)

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.errorbar(k_values, y_values, yerr=errors)
plt.xlabel("k value (# of centroids)")
plt.ylabel("SSE")
plt.title("max_iter = " + str(max_iter))
plt.xticks(k_values)
plt.show()
