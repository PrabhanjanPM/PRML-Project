
#Bivariate case

import matplotlib
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


################################## MODEL ###################################

#Create the grid to plot
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)

#Model data as gaussians in format (X, Y, sig_x, sig_y, mu_x, mu_y, mu_xy)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)

# difference of Gaussians
Z = 10.0 * (Z2 - Z1)

################################### PLOT ###################################

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS)
plt.title('GMM Plot')

plt.show()

############################################################################