# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 08:24:00 2017

@author: Gautam
"""
import numpy as np
import csv
import networkx as nx
from scipy.optimize import minimize
import math

def parseDataFile():
    mat = np.zeros([4032,4032])
    with open('C:\Users\Gautam\Desktop\grad-books\comp652\project\\facebook_combined.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if(int(row[1])<=4031):
                mat[int(row[0]),int(row[1])]=1
    return mat
#problem 1 : we want to study how large the largest clique is    
#randomly sample from the facebook set to generate test and training data.
#problem we want to study how to generate networks that accurately simulate real world networks
def fisher_yates_shuffle(probMatrix):
    out = np.array(probMatrix, copy='true')
    rows = probMatrix.shape[0]
    i = rows-1
    while(i>=1):
        j = np.random.randint(i)
        out = swap_nodes_2(out,i, j)
        i=i-1
    return out
train = parseDataFile()[0:2048,0:2048]
#in order to use random sampling uncomment the below line, it takes a while
#train = fisher_yates_shuffle(parseDataFile())[0:2048,0:2048]

def getProbMatrix(n,p):
    p=p.reshape(2,2)
    #p=p.reshape(3,3)
    X=np.kron(p,p)
    while (n>1):
        X = np.kron(X,p)
        n=n-1
    return X    

#randomly permute the values
    
def swap_nodes_2(probMatrix, i, j):
    probMatrix2 = np.array(probMatrix, copy='true')
    rows = probMatrix.shape[0]
    swapped = np.zeros(probMatrix.shape)
    probMatrix2[j,j] = probMatrix[i,i]
    probMatrix2[i,i] = probMatrix[j,j]
    swapped[i,i]=1
    swapped[j,j]=1
    probMatrix2[i,j] = probMatrix[j,i]  
    probMatrix2[j,i] = probMatrix[i,j]
    swapped[i,i]=1
    swapped[i,j]=1
    swapped[j,i]=1
    for x in range(0, rows):
        if(swapped[i,x]==0):
            probMatrix2[i,x] = probMatrix[j,x]
            probMatrix2[j,x] = probMatrix[i,x]
            swapped[i,x] =1
            swapped[j,x] =1
    for x in range(0, rows):
        if(swapped[x,i]==0):
            probMatrix2[x,i] = probMatrix[x,j]
            probMatrix2[x,j] = probMatrix[x,i]
            swapped[x,i] =1
            swapped[x,j] =1
    return probMatrix2



def random_permutation(probMatrix):
    out = np.array(probMatrix, copy='true')
    n=20
    while n>1:
        rows = probMatrix.shape[0]
        i = np.random.randint(rows)
        j = np.random.randint(rows)
        probMatrix2, diff = swap_nodes(probMatrix,i,j,train)
        r = np.random.uniform()
        #print 'diff is '+ str(diff)
        #print 'threshold is' + str(np.log(r))
        if(diff) < np.log(r):
            out = probMatrix
            return out
        else:
            probMatrix = probMatrix2
        n=n-1
    return out


def swap_nodes(probMatrix, i, j, train):
    diff=0
    probMatrix2 = np.array(probMatrix, copy='true')
    rows = probMatrix.shape[0]
    swapped = np.zeros(probMatrix.shape)
    probMatrix2[j,j] = probMatrix[i,i]
    probMatrix2[i,i] = probMatrix[j,j]
    diff = diff - (bLike(train[i,i],probMatrix[i,i]) + bLike(train[j,j],probMatrix[j,j]) - bLike(train[j,j],probMatrix2[j,j]) - bLike(train[j,j],probMatrix2[j,j]))
    swapped[i,i]=1
    swapped[j,j]=1
    probMatrix2[i,j] = probMatrix[j,i]  
    probMatrix2[j,i] = probMatrix[i,j]
    diff = diff - (bLike(train[i,j],probMatrix[i,j]) + bLike(train[j,i],probMatrix[j,i]) - bLike(train[j,i],probMatrix2[j,i]) - bLike(train[i,j],probMatrix2[i,j]))
    swapped[i,i]=1
    swapped[i,j]=1
    swapped[j,i]=1
    for x in range(0, rows):
        if(swapped[i,x]==0):
            probMatrix2[i,x] = probMatrix[j,x]
            probMatrix2[j,x] = probMatrix[i,x]
            diff = diff - (bLike(train[i,x],probMatrix[i,x]) + bLike(train[j,x],probMatrix[j,x]) - bLike(train[j,x],probMatrix2[j,x]) - bLike(train[i,x],probMatrix2[i,x]))
            swapped[i,x] =1
            swapped[j,x] =1
    for x in range(0, rows):
        if(swapped[x,i]==0):
            probMatrix2[x,i] = probMatrix[x,j]
            probMatrix2[x,j] = probMatrix[x,i]
            diff = diff - (bLike(train[x,i],probMatrix[x,i]) + bLike(train[x,j],probMatrix[x,j]) - bLike(train[x,j],probMatrix2[x,j]) - bLike(train[x,i],probMatrix2[x,i]))
            swapped[x,i] =1
            swapped[x,j] =1
    return (probMatrix2,diff)

def bLike(x,p):
    if(x==1):
        if(p==0):
            return -100
        else:
            return p
    else:
        if(p==1):
            return -100
        else:
            return 1-p

def generateMatrixFromProb(probMatrix):
    rows = probMatrix.shape[0]
    mat=np.zeros([rows,rows])
    for x in range(0, rows):
        for y in range(0, rows):
            if(x!=y):
                r = np.random.uniform()
                if(r<probMatrix[x,y]):
                    mat[x,y]=1
    return mat
#log likelihood    
def likelihood(adj,mat):
    rows = mat.shape[0]
    outval = 0
    for x in range(0, rows):
        for y in range(0, rows):
            if(mat[x,y]==1):
                outval = outval + np.log(adj[x,y]+ 0.00000001)
            else:
                outval = outval + np.log(1-adj[x,y] +0.00000001)
    return outval
#lieklihood of empty graph, taylor series estimate
def likelihood0Estimate(k,p):
    return - np.sum(p)**k - 0.5*np.sum(np.square(p))**k
    
def likelihood1Estimate(theta, train):
    loc = np.transpose(np.nonzero(train))
    outval = 0
    for i in range(0, len(loc)):
        elem = theta[loc[i,0], loc[i,1]]
        outval = outval + np.log(elem+0.00000001) - np.log(1-elem+0.00000001)
    return outval

def likelihoodEstimate(theta, train, p, k):
    return likelihood0Estimate(k,p) + likelihood1Estimate(theta,train)

def loss(p):
    print 'p is'+str(p)
    theta = getProbMatrix(10,p)
    l=0
    for i in range(0,100):
        l+=likelihoodEstimate(random_permutation(theta),train, p, 11)
    print 'MC estimated Log Likelihood '+str(l/100)
    return -l/100
    #return np.linalg.norm(theta-train)

def gradient(p0,epsilon, orig_loss):
    grad = np.zeros([len(p0)])
    for i in range(len(p0)):
        p1 = np.array(p0, copy = 'true')
        p1[i] = p1[i]+epsilon
        grad[i] = (loss(p1) - orig_loss)/ epsilon 
    return grad
    
def gradient_descent(p0, steps, epsilon, step_size,threshold):
    orig_loss = loss(p0)
    while(steps>0):
        grad = gradient(p0,epsilon,orig_loss)
        p1 = p0 + np.dot(step_size,grad)
        print str(grad)
        new_loss =  loss(p1)
        if(np.abs(new_loss - orig_loss)<threshold):
            return p1
        orig_loss = new_loss
        p0=p1
        steps = steps-1
    return p1

def construct_jacobian(func,epsilon):
    def jac(x, *args):
        x0 = np.asfarray(x)
        f0 = np.atleast_1d(func(*((x0,)+args)))
        jac = np.zeros([len(x0),len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0

        return jac.transpose()
    return jac

def generateProbMatrix2(k,p,size):
    p=p.reshape(size,size)
    rows = size**k
    mat = np.zeros([rows, rows])
    for x in range(0, rows):
        for y in range(0, rows):
            if(x!=y):
                prod=1
                for i in range(0,k):
                    counter = int((math.floor((x)/(size**i)))%size)
                    counter2 = int((math.floor((y)/(size**i)))%size)
                    prod = prod*p[counter, counter2]
                mat[x,y] = prod
    return mat
#to run the optimization, uncomment the below line    
#res=minimize(loss,np.random.rand(4),  bounds=[(0.1,0.9),(0.1,0.9),(0.1,0.9),(0.1, 0.9)], method='TNC', jac=construct_jacobian(loss, 0.01)) 