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

def differentiate(error, data):
    n = data.shape[0]
    derivative = (1.0/n)*np.dot(np.transpose(data), error)
    return derivative

def gradient_descent(data, label, model, learning_rate, iterations):
    #print "Starting gradient descent- "
    for k in range(0, iterations):
        theta_x = np.dot(data, model)
        h_theta = activation(theta_x)
        cost    = cost_function(label, h_theta)
        error   = h_theta - label 
        derivative = differentiate(error, data)
        model   = model - learning_rate*derivative
        #print "Cost in iteration " + str(k) + " - " + str(cost)
    return model 

def decide(data, label, model):
    n        = label.shape[0]
    decision = np.zeros(n)  
    theta_x  = np.dot(data, model)
    h_theta  = activation(theta_x)
    for i in range(0,n):
        if(h_theta[i]>=0.4):
            decision[i] = 1
    return decision
    
data_   = np.genfromtxt("../features/data", delimiter= ",")
n      = data_.shape[0]
data_   = np.transpose(np.vstack((np.ones(n),data_)))
label_  = np.append(np.zeros(1508), np.ones(1508))
step    = 150
costs   = []
for k in range(1,11):
    if(k!=10):
        data1 = data_[0:step*k,:]
        data2 = data_[1508:1508+step*k,:]
        label1 = label_[0:step*k]
        label2 = label_[1508:1508+step*k]
        data  = np.vstack((data1,data2))
        n = data.shape[0]
        label  = np.hstack((label1,label2))
        print "Using "+str(2*k*step)+"  samples"
    else:
        data = data_
        label = label_
        n = data.shape[0]
        print "Using all samples"

    model  = np.array([2, -2])
    model  = gradient_descent(data, label, model, 0.1, 1000 )
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
    print "model - ", model
    np.savetxt("model"+str(k),model)

np.savetxt("train_costs", costs)
