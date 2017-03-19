import numpy as np
import math 

from matplotlib import pyplot as plot


def probability(x,mean,inv_cov,C):
    #d = x.shape[0]
    d = 1
    two_pi = 2*math.pi
    const  = math.pow(two_pi,d)
    det    = np.prod(two_pi*C)
    x = x-mean
    argument = np.dot(x,np.dot(inv_cov,x))
    if(const == 0):
        print det 
        print C
    probability = (math.exp(-0.5*argument))/math.sqrt(const)
    if(probability>1):
        print "Error"
        print "X- " + str(x)
        print "mean - " + str(mean)
        print "inv - "
        print inv_cov
        print "det - " + str(det)
        exit()
    return probability


data = np.genfromtxt("../features/test_data",delimiter=",")
#d = data.shape[1]
d = 1

n = data.shape[0]
data = data
mean1 = np.loadtxt("mean1")
mean2 = np.loadtxt("mean2")
one   = np.ones((d,))

#For general n dimensional data
#cov1  = np.loadtxt("cov1")
#cov2  = np.loadtxt("cov2")

#For 0d covariance
cov1  = [[np.loadtxt("cov1")]]
cov2  = [[np.loadtxt("cov2")]]
C1    = np.diag(cov1)
C2    = np.diag(cov2)
inv1  = np.diag(one/C1)
inv2  = np.diag(one/C2)
w1    = np.loadtxt("w1")
w2    = np.loadtxt("w2")

l = 0
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
correct = 0
for x in data:
    probability1 = probability(x,mean1,inv1,C1)
    probability2 = probability(x,mean2,inv2,C2)
    total_probability = w1*probability1 + w2*probability2
    if(total_probability == 0):
        zeros = zeros+1
        print "Zeros - " + str(zeros)
        continue  
    likelihood1  = (w1*probability1)/total_probability
    likelihood2  = (w2*probability2)/total_probability
    #if(likelihood1>likelihood2):
    #    print "Class1"
    #else:
    #    print "Class2"
    #modify input to use labels. modify code to process it

    if(likelihood1>likelihood2):
        if(l<760):
            true_positive = true_positive + 1
            correct = correct+1
        else:
            false_positive = false_positive + 1
    else:
        if(l<760):
            false_negative = false_negative + 1
        else:
            true_negative = true_negative + 1
            correct = correct + 1
    l = l+1
precision = float(true_positive)/(true_positive+false_positive)
accuracy  = float(true_positive + true_negative)/1710
print "Confusion Matrix - "
print " | "+ str(true_positive) +" | "+  str(false_negative) + " | "
print " | "+ str(false_positive) +" | "+  str(true_negative) + " | "
print "Accuracy - " + str(accuracy)
print "Precision - "+ str(precision)
print correct

precision = float(true_positive)/(true_positive+false_positive)
accuracy  = float(true_positive + true_negative)/1710
print "Confusion Matrix - "
print " | "+ str(true_positive) +" | "+  str(false_negative) + " | "
print " | "+ str(false_positive) +" | "+  str(true_negative) + " | "
print "Accuracy - " + str(accuracy)
print "Precision - "+ str(precision)
plot.scatter(data[0:760],-100*np.ones(760),c="red",s=np.pi*25,label="genuine")
plot.scatter(data[760:1710],100*np.ones(950),c="blue",s=np.pi*25,label="spoofed")
plot.legend(loc=2)
plot.show()
