import numpy as np 
import math 

from matplotlib import pyplot as plt 

def sigmoid(x):
    n = x.shape[0]
    return (np.ones(n)/(np.ones(n) + np.exp(-x)))

def activation(x):
    n = x.shape[0]
    h_theta  = sigmoid(x)
    return h_theta

def cost_function(label, h_theta):
    #Can this be vectorized?
    n = label.shape[0]
    foreach_cost = []
    for i in range(0, len(label)):
        if(label[i] == 1):
            foreach_cost.append(-np.log(h_theta[i]))
        else:
            foreach_cost.append(-np.log(1 - h_theta[i]))
    cost = (1.0/n)*np.sum(np.array(foreach_cost))
    return cost 

def decide(data, label, model):
    n        = label.shape[0]
    decision = np.zeros(n) 
    theta_x  = np.dot(data, model)
    h_theta  = activation(theta_x)
    for i in range(0,n):
        if(h_theta[i]>=0.4):
            decision[i] = 1
    return decision
    
data   = np.genfromtxt("../features/test_data", delimiter= ",")
n      = data.shape[0]
data   = np.transpose(np.vstack((np.ones(n),data)))
label  = np.append(np.zeros(760), np.ones(950))
step   = 150
costs  = []
for k in range(1,11):
    model = np.loadtxt("model"+str(k))
    print "Using model -", model 
    print "Model trained with "+str(2*k*step)+" samples"
    decision = decide(data, label, model)
    theta_x = np.dot(data, model)
    h_theta = activation(theta_x)
    costs.append(cost_function(label, h_theta))
    
    true_positive  = 0.0 
    true_negative  = 0.0 
    false_positive = 0.0
    false_negative = 0.0

    for i in range(0,n):
        if(decision[i] == 1):
            if(label[i] == 0):
                false_positive = false_positive + 1 
            else:
                true_positive   = true_positive  + 1
        else:
            if(label[i] == 0):
                true_negative = true_negative + 1 
            else:
                false_negative = false_negative + 1 

    print "true positive- ", true_positive  
    print "true negative- ",  true_negative  
    print "false positive- ", false_positive 
    print "false negative- ", false_negative 

    accuracy  = (true_positive+true_negative)/n
    precision = (true_positive)/(true_positive+false_positive)
    recall    = (true_positive)/(true_positive+false_negative)
    print "Accuracy- ", accuracy
    print "Precision- ", precision
    print "Recall- ", recall 
    print 

np.savetxt("test_costs", costs)
