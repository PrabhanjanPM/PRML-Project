import numpy as np 

from matplotlib import pyplot as plt 

step = 150
number_of_samples = step*np.array([1,2,3,4,5,6,7,8,9,10])
train_costs = np.loadtxt("train_costs")
test_costs  = np.loadtxt("test_costs")
plt.plot(number_of_samples, train_costs)
plt.plot(number_of_samples, test_costs)
plt.show()

