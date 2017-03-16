import numpy as np 
import math 

def probability(x,mean,inv_covariance,covariance):
    d = x.shape
    xt_C_inv = np.inner(x-mean,inv_covariance)
    argument = np.inner(xt_C_inv,x-mean)
    determinant = np.prod(np.diag(covariance))
    two_pi = 2*math.pi
    const = np.power(two_pi,d)*determinant
    probability = np.exp(-0.5*argument)/math.sqrt(const)
    return probability

def find_covariance(d,n,X,mean):
    covariance = np.outer(np.zeros(d),np.zeros(d))
    #print covariance
    for i in range(0,n):
        sample = X[i,:]-mean
        covariance = covariance + np.outer(sample,sample)
    covariance = covariance/(n*np.ones((d,d)))
    return covariance
        

data = np.genfromtxt("data", delimiter=',')
data1 = np.genfromtxt("data1", delimiter=',')
data2 = np.genfromtxt("data2", delimiter=',')
n = data.shape[0]
d = data.shape[1]

#mean1_list = []
#mean2_list = []

#for i in range(0,d):
#    mean1_list.append(np.mean(data[0:n/2,i]))
#    mean2_list.append(np.mean(data[n/2:n-1,i]))

mean1 = data[0,:]
mean2 = data[1,:]
#mean1  = np.array(mean1_list)
#mean2  = np.array(mean2_list)
#print "Mean1 - " + str(mean1)
#print "Mean2 - " + str(mean2)

C1  = np.ones(d)
C2  = np.ones(d)
one = np.ones(d)
covariance1 = np.diag(C1)
covariance2 = np.diag(C2)
inv_covariance1 = np.diag(one/C1)
inv_covariance2 = np.diag(one/C2)

#print "Covariance1 - \n" + str(covariance1)
#print "Covariance2 - \n" + str(covariance2)
#print

weight1 = 0.5
weight2 = 0.5

#Check some condition to set the number of iterations instead of constant 100 
for i in range(0,100):
    #Expectation 
    n1 = 0
    n2 = 0
    X1 = []
    X2 = []
    for x in data:
        probability1 = probability(x, mean1, inv_covariance1, covariance1)
        probability2 = probability(x, mean2, inv_covariance2, covariance2)
        probability_all  = weight1*probability1 + weight2*probability2
        likelihood1  = weight1*probability1/probability_all
        likelihood2  = weight2*probability2/probability_all
        #print "Likelihood of class 1 - " + str(likelihood1)
        #print "Likelihood of class 2 - " + str(likelihood2)
        #print 
        if(likelihood1>likelihood2):
            n1 = n1+1
            if(X1 == []):
                X1 = x
            else:
                X1 = np.vstack((X1,x))
        else:
            n2 = n2+1 
            if(X2 == []):
                X2 = x
            else:
                X2 = np.vstack((X2,x))

    #Maximization
    mean1_list = []
    mean2_list = []
    for i in range(0,d):
        mean1_list.append(np.mean(X1[:,i]))
        mean2_list.append(np.mean(X2[:,i]))

    mean1  = np.array(mean1_list)
    mean2  = np.array(mean2_list)

    weight1 = float(n1)/(n1+n2)
    weight2 = float(n2)/(n1+n2)
    C1 = np.diag(find_covariance(d,n1,X1,mean1))
    C2 = np.diag(find_covariance(d,n2,X2,mean2))
    covariance1 = np.diag(C1)
    covariance2 = np.diag(C2)
    inv_covariance1 = np.diag(one/C1)
    inv_covariance2 = np.diag(one/C2)

print "Mean of class1 - " + str(mean1)
print "Covariance of class1 - " 
print covariance1
print "Mean of class2 - " + str(mean2)
print "Covariance of class2 - "
print covariance2
