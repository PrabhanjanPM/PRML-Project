import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

data1 = np.genfromtxt("../features/data",delimiter=',')
m=data1.size
b=np.ones(m)
c=np.append(np.ones(1508),np.zeros(1508))
#plt.plot(data1,c,'ro')
#plt.show()

k=np.vstack([b,data1,c])
x=np.transpose(k)
np.random.shuffle(x)
X1=x[:,[0,1]]
Y1=x[:,2]
theta=[1,1]

alpha=0.3
number_of_iterations=50


def Sigmoid(z):
	d = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return d

def Hypothesis(theta, x):
	z = 0
	z = np.inner(x,theta)
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	error_sum = 0
	for i in range(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		error_sum += error
	const = -1/m
	J = const * error_sum
	#print 'cost is ', J 
	return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		deri = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - deri
		new_theta.append(new_theta_value)
	return new_theta

cost=np.zeros(number_of_iterations)

def error(X,Y,theta,m):
	#theta1=[]
	c=0
	for i in range(0,number_of_iterations):
		cost[i]=Cost_Function(X,Y,theta,m)
		theta=Gradient_Descent(X,Y,theta,m,alpha)
		#theta1.append(theta)
		c=c+1
	return cost[c-1],theta


err=np.zeros(15)
err_validate=np.zeros(15)
c=0
validation_x=X1[1500:3000]
validation_y=Y1[1500:3000]
for k in range(1,16):
	i=k*100
	X=X1[0:i]
	Y=Y1[0:i]
	m=np.size(Y)
	err[c],theta=error(X,Y,theta,m)
        print "model- ", theta
        np.savetxt("mega_model"+str(k), theta)
	err_validate[c]=Cost_Function(validation_x,validation_y,theta,np.size(validation_y))
	#err_validate[c],theta=error(validation_x,validation_y,theta,np.size(validation_y))
	c=c+1
print err,err_validate
plt.plot(np.arange(0,1500,100),err, label='training curve')
plt.plot(np.arange(0,1500,100),err_validate, label='testing curve')
plt.legend()
plt.show()




