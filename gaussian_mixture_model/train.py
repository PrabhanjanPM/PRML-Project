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

def find_covariance(data,mean,px,n1):
    #d = data.shape[1]
    d = 1 
    n = data.shape[0]
    cov = np.outer(np.zeros((d,)),np.zeros((d,)))
    for i in range(0,n):
        sample = data[i] - mean
        cov    = cov + np.outer(sample,sample)*px[i]
    cov = cov/n1
    #print cov
    diag_cov = np.diag(np.diag(cov))
    return diag_cov

data = np.genfromtxt("../features/data",delimiter=",")
#d = data.shape[1]
d = 1
n = data.shape[0]
mean1 = (data[0]+data[188])/2
mean2 = (data[2002]+data[1877])/2
one   = np.ones((d,))
cov1  = np.eye(d)*0.0001
cov2  = np.eye(d)*0.0001
C1    = np.diag(cov1)
C2    = np.diag(cov2)
inv1  = np.diag(one/C1)
inv2  = np.diag(one/C2)
w1    = 0.5
w2    = 0.5
for i in range(0,50):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    class1_false = 0
    class1_missed = 0
    correct = 0
    zeros = 0
    p1X = list([])
    p2X = list([])
    l = 0
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
        if(likelihood1>likelihood2):
            if(l<1508):
                true_positive = true_positive + 1
            else:
                false_positive = false_positive + 1
        if(likelihood2>likelihood1):
            if(l<1508):
                false_negative = false_negative + 1
            else:
                true_negative = true_negative + 1
        if(likelihood1>likelihood2 and l>1507):
            class1_false = class1_false +1 
        if(likelihood2>likelihood1 and l<=1507):
            class1_missed = class1_missed + 1 
        if(likelihood1>likelihood2 and l<=1507):
            correct = correct+1
        if(likelihood1<likelihood2 and l>1507):
            correct = correct+1
        l = l+1
        p1X.append(likelihood1)
        p2X.append(likelihood2)

    n1    = sum(p1X)
    w1    = n1/n
    n2    = sum(p2X)
    w2    = n2/n
    mean1 = np.zeros((d,))
    mean2 = np.zeros((d,))
    i = 0
    for x in data:
        mean1 = mean1 + p1X[i]*x
        mean2 = mean2 + p2X[i]*x
        i = i+1 
    mean1 = mean1/n1
    mean2 = mean2/n2
    cov1  = find_covariance(data,mean1,p1X,n1)
    cov2  = find_covariance(data,mean2,p2X,n2)
    #print "class1 false - " + str(class1_false)
    #print "class1 missed - " + str(class1_missed)
    #print correct
print "Mean1 (genuine)- "+ str(mean1)
print "Covariance1 - " + str(cov1)
print "Weight1 - " + str(w1)
print "Mean2 (spoofed)- "+ str(mean2)
print "Covariance2 - " + str(cov2)
print "Weight2 - " + str(w2)

np.savetxt("mean1",mean1)
np.savetxt("mean2",mean2)
np.savetxt("cov1",cov1)
np.savetxt("cov2",cov2)
np.savetxt("w1",[w1])
np.savetxt("w2",[w2])

precision = float(true_positive)/(true_positive+false_positive)
accuracy  = float(true_positive + true_negative)/3016
print "Confusion Matrix - "
print " | "+ str(true_positive) +" | "+  str(false_negative) + " | "
print " | "+ str(false_positive) +" | "+  str(true_negative) + " | "
print "Accuracy - " + str(accuracy)
print "Precision - "+ str(precision)


plot.scatter(data[0:1508],-100*np.ones(1508,), c="red", s=np.pi*25,label="genuine")
plot.scatter(data[1508:3016],100*np.ones(1508,), c="blue", s=np.pi*25,label="spoofed")
plot.scatter(np.hstack((mean1,mean2)),np.hstack((-100,100)),s=np.pi*40,c="black",label="mean")
plot.legend(loc=2)
plot.show()

