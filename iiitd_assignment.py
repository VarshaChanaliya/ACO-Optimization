#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
from geopy.distance import geodesic


# In[114]:


data = pd.read_csv("addresses.csv")
len(data)


# In[115]:


base_address=(12.919716, 77.645059)
threshold_distance=3
distance = []


# In[116]:


for i in range(0,len(data)):
    data_address = data.coordinates[i]
    value = geodesic(data_address,base_address).kilometers
    distance.insert(i,value)
    


# In[ ]:





# In[98]:


data.loc[:,'distance_from_base_addrs'] = distance
list = []
for x in range (0,len(data)):
    if (data.distance_from_base_addrs[x]< threshold_distance):
        list.insert(x,'distance less than threshold')
    else: list.insert(x,'distance greater than threshold')

data.loc[:,'label'] = list
data = data.set_index("label")
data = data.drop("distance less than threshold", axis=0)


# # Output (with removed addresses with distance from base addrs greater than 3kms)

# In[99]:


data.head(10)


# In[384]:


list = data.coordinates.values
lat = []
for i in range(0,len(data)):
              lat.append(float(list[i].split(',')[0]))

lon = []
for i in range(0,len(data)):
              lon.append(float(list[i].split(',')[1]))
X = []
for i in range(0,len(data)):
    X.append((lat[i],lon[i]))


# In[391]:


data.loc[:,'lat'] = lat
data.loc[:,'long'] = lon
data.loc[:,'nodes'] = range(0,len(data))
data.head()


# In[370]:


l = []
for i in X:
    if i not in l:
         l.append(i)
len(X)            


# In[348]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
plt.scatter(a,b,s=14)
plt.show()


# In[ ]:





# In[349]:


clf = KMeans(n_clusters = 2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors = ["g.","r.","b.","c.","k.","o."]
for i in range (0,len(data)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:,0], centroids[:,1], marker = '+', s = 150, linewidths = 5)
plt.show()


# In[350]:


import scipy
dist = scipy.spatial.distance.pdist(X, metric='euclidean')
dist_mat = scipy.spatial.distance.squareform(dist, force='tomatrix', checks=True)
print(dist_mat)


# In[371]:


import pants
import math
import random


# In[372]:


def euclidean(a, b):
    return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))


# In[394]:


world = pants.World(l, euclidean,N = 150, L = 150 , A = 2, B = 3)


# In[395]:


solver = pants.Solver()


# In[396]:


solution = solver.solve(world)
# or
solutions = solver.solutions(world)


# In[397]:


print(solution.distance)


# In[ ]:





# In[398]:


best = float("inf")
for solution in solutions:
  assert solution.distance < best
  best = solution.distance


# In[399]:


tour = solution.tour


# In[ ]:





# In[400]:


path = data.set_index(['lat','long'])['nodes'].loc[tour].tolist()


# In[ ]:





# In[ ]:




