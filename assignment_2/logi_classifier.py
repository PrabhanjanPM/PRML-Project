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
X=x[:,[0,1]]
Y=x[:,2]
theta=[-1.5,-1.5]

alpha=0.5
number_of_iterations=500

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
	print 'cost is ', J 
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

for i in range(0,number_of_iterations):
	cost[i]=Cost_Function(X,Y,theta,m)
	theta=Gradient_Descent(X,Y,theta,m,alpha)

print theta
reg=theta[0]+theta[1]*np.array(x[:,1])

plt.plot(cost)
plt.show()

data1 = np.genfromtxt("data",delimiter=',')
m=data1.size
b=np.ones(m)
c=np.append(np.zeros(1508),np.ones(1508))
plt.plot(data1,c,'ro')
plt.plot(x[:,1],reg)
#plt.show()

data = np.genfromtxt("test_data",delimiter=',')
m=data.size
b=np.ones(m)
k=np.vstack([b,data])
X=np.transpose(k)
c1=np.append(np.zeros(760),np.ones(950))
#plt.plot(data,c1,'bs')
plt.show()

p=np.zeros(m)
def predict(theta, X):
	for i in range(m):
		xi = X[i]
		theta=np.array(theta)
		pro=np.dot(xi,theta)
		res=Sigmoid(pro)

		if pro> 0:
            		p[i] = 0
        	else:
           		p[i] = 1
	return p


s=predict(theta,X)
tn=0
tp=0
fn=0
fp=0

for i in range(m):
	if(i<760):
		if(s[i]==0):
			tp=tp+1
		else:
			fn=fn+1
	else:
		if(s[i]==1):
			tn=tn+1
		else:
			fp=fp+1



#plt.plot(data,s,'ro')
plt.plot(data,c1,'bs')
plt.plot(x[:,1],reg)
plt.show()
acc=float((tp+tn)/float(tp+tn+fp+fn))
print s,acc
print tp,fn,tn,fp

