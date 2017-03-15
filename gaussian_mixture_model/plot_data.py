#Supports only 2d plots 
#Usage - python plot_data.py

import numpy as np

from matplotlib import pyplot as plot 

data1 = np.genfromtxt("data1",delimiter=',')
data2 = np.genfromtxt("data2",delimiter=',')
data  = np.genfromtxt("data" ,delimiter=',')

plot.scatter(data1[:,0], data1[:,1])
plot.scatter(data2[:,0], data2[:,1])
plot.show()

plot.scatter(data[:,0], data[:,1])
plot.show()



