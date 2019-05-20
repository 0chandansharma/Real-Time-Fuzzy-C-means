# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:58:25 2019

@author: chandan_sharma
"""

import pandas as pd
import numpy as np
import random
import operator
import math
import time
from sklearn import metrics
#df_full = pd.read_csv(r"C:\Users\chandan sharma\Desktop\SUSY\SUSY.csv")
#df_full=df_full.iloc[0:20000]
df_full = pd.read_csv(r"C:\Users\chandan sharma\Desktop\research\majorproject\ppr1\code\python\wine\wine.csv")
#df_full = pd.read_csv(r"C:\Users\chandan sharma\Desktop\research\majorproject\ppr1\code\python\dataset_12_mfeat-factors.csv")
#df_full = pd.read_csv(r"C:\Users\chandan sharma\Desktop\research\majorproject\ppr1\code\python\KDD\KDDCup99.csv")

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
#df_full['label']=number.fit_transform(df_full['label'].astype('str'))
#df_full['protocol_type']=number.fit_transform(df_full['protocol_type'].astype('str'))
#df_full['service']=number.fit_transform(df_full['service'].astype('str'))
#df_full['flag']=number.fit_transform(df_full['flag'].astype('str'))
#df_full['class']=number.fit_transform(df_full['class'].astype('str'))
columns = list(df_full.columns)
features = columns[:len(columns)-1]
class_labels = list(df_full[columns[-1]])
df = df_full[features]
num_attr = len(features)
# cluster number
#k = 2
x=[];
y=[]
chunks_size= 2
# max number of iterations

def myval(membership_mat,sx):
    global Dm,g,myv;
    M=[]
    D=[]
    for j in range(k):
        for i in range(len(sx)):
            g+=membership_mat[i][j];
        M.append(g/len(sx))
        g=0
    #print(M)
    for i in range(k):
        for j in range(k):
            if i !=j:
               Dm=M[i]-M[j];
               Dm=math.pow(Dm,2);
               myv+=Dm
               D.append(Dm);
    #print(D)          
    #for i in range(D):
     #   sum+=D[i]
    #print(sum(D))
    #print(myv)
    return myv
                
            
def vp(membership_mat,sx):
    global vpp
    max1=0
    for i in range(len(sx)):
        for j in range(k):
               
            if membership_mat[i][j]>max1:
                max1=membership_mat[i][j];
            else:
                max1=max1;
        vpp+=max1
        #print("________________________",max1)
        #vpp+=f
        nr=vpp
    result= nr/n    
    return result
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
def mek(membership_mat,sx):
    global sm
    for j in range(k):
        for i in range(len(sx)):
            sm+= membership_mat[i][j]
    #mk=sm/n;
    return sm;        
             
    
def partitioncoff(membership_mat,sx):
    global pc
    for j in range(k):
        for i in range(len(sx)):
            pc+=(membership_mat[i][j]*membership_mat[i][j])
        #print("_____",pc)    
    pc= pc/len(sx);
    return pc        
def entropy(membership_mat,sx):
    global ent
    #count=0;
    for j in range(k):
        for i in range(len(sx)):
            g=math.pow(membership_mat[i][j],2)
            ent+= g*(math.log(membership_mat[i][j]))
    res=-(ent)/len(sx)
    return res
    #print("_________")
def fuzzyentropy(membership_mat,sx):
    global ent
    #count=0;
    for j in range(k):
        for i in range(len(sx)):
            a=(membership_mat[i][j])
            b=1-membership_mat[i][j]
            ent+= a*(math.log(membership_mat[i][j]))+b*(math.log(1-membership_mat[i][j]))
            
    res=-(ent)/len(sx)
    return res
    #print("_________")

def initializeMembershipMatrix(x):
    membership_mat = list()
    for i in range(len(x)):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x1/summation for x1 in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def initializeWeight(x):
    Weight = [1 for i in range(len(x))]
    return Weight


def calculateClusterCenter(membership_mat, W, sx):
    #cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
        x = list(zip(*membership_mat))[j]
        xraised = [e ** m for e in x]
        xraised_mul_W = [a*b for a, b in zip(xraised, W)]
        denominator = sum(xraised_mul_W)
        temp_num = list()
        for i in range(len(sx)):
            data_point = sx[i]
            prod = [xraised_mul_W[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def updatemembershipvalue(U, C, sx):
    alpha = float(1/(m-1))
    for i in range(k):
        for j in range(len(sx)):
            x = sx[j]
            numerator = [(a-b)**2 for a, b in zip(x, C[i])]
            num = sum(numerator)
            dis = [map(operator.sub, x, C[k1]) for k1 in range(k)]
            denominator = [map(lambda x: x**2, dis[j1]) for j1 in range(k)]
            den = [sum(denominator[k1]) for k1 in range(k)]
            res = sum([math.pow(float(num/den[k1]), alpha) for k1 in range(k)])
            U[j][i] = float(1/res)
    return U
def getClusters(membership_mat,sx):
    cluster_labels = list()
    for i in range(len(sx)):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

def updateweight(U1, c):
    W = list()
    for j in range(c):
        w1 = list()
        u = list(zip(*U1[j]))
        u=list(u)
        for i in range(k):
            u1 = sum(u[i])
            w1.append(u1)
        W.append(w1)
    W1 = list()
    W1 = W[0]
    for j in range(1, len(W)):
        W1 = W1+W[j]
    return W1


def WFCM(sx, W, U, C):
    i = 0
    while(i <= max_iter):
        U = updatemembershipvalue(U, C, sx)
        C = calculateClusterCenter(U, W, sx)
        cluster_labels = getClusters(U,sx)
        i += 1
    #global calentropy
    #entro=entropy(U, sx)
    #fuzzyent=fuzzyentropy(U)
    #pcvalue=partitioncoff(U,sx) 
    #dmk=mek(U)
    #findvp=vp(U,sx)
    #myres=myval(U,sx)
    #calentropy= entro+myres
    #print(membership_mat)
    #print("datasize=",len(sx))
    ##print("num of cluster=",k)
    #print("entro=",entro) 
    #print("Fuzzy entropy__",fuzzyent)
   # print("PC=",pcvalue)
    ##print("DMK value= ",dmk)
    #print("find vp=",findvp)
    #print("saparation",myres)
    #print("RESULT1_________MIN________________________",entro+myres)
    return C, U,cluster_labels

def WFCM2(sx, W, U, C):
    i = 0
    while(i <= max_iter):
        U = updatemembershipvalue(U, C, sx)
        C = calculateClusterCenter(U, W, sx)
        cluster_labels = getClusters(U,sx)
        i += 1
    global calentropy
    entro=entropy(U, sx)
    fuzzyent=fuzzyentropy(U, sx)
    pcvalue=partitioncoff(U, sx) 
    #dmk=mek(U)
    #findvp=vp(U,sx)
    myres=myval(U,sx)
    calentropy= entro+myres
    #print(membership_mat)
    #print("datasize=",len(sx))
    print("num of cluster=",k)
    print("entro=",entro) 
    print("Fuzzy entropy__",fuzzyent)
    print("PC=",pcvalue)
    #print("DMK value= ",dmk)
    #p#rint("find vp=",findvp)
    #p#rint("saparation",myres)
    print("RESULT1_________MIN________________________",fuzzyent+myres)
    y.append(fuzzyent+myres)
    #print("silhoutte  score::::",metrics.silhouette_score(sx,cluster_labels,metric='euclidean'))
    #print("result_2_____________________________________________",sil)
    #print("____________________Calinski-harabaz index: ", (metrics.calinski_harabaz_score(sx,cluster_labels)))
    return C, U,cluster_labels

def oFCM(p):
    X_sampled = list()
    for j in range(chunks_size):
        l = (len(df)//chunks_size)
        x = list()
        for i in range(j*l, min(j*l+l, len(df))):
            data = list(df.iloc[i])
            x.append(data)
        X_sampled.append(x)
    U1 = list()
    C1 = list()
    W = initializeWeight(X_sampled[0])
    U = initializeMembershipMatrix(X_sampled[0])
    center = calculateClusterCenter(U, W, X_sampled[0])
    C, U,labels = WFCM(X_sampled[0], W, U, center)
    #print(C)
    #print("+++++")
    #print(U)
    #print("+++++")
    U1.append(U)
    C1.append(C)    
    for j in range(1, chunks_size):
        X = X_sampled[j]
        W = initializeWeight(X)
        U = initializeMembershipMatrix(X)
        U = updatemembershipvalue(U, C1[j-1], X)
        center = calculateClusterCenter(U, W, X)
        #print("hi")
        C, U,labels = WFCM(X, W, U, center)
        U1.append(U)
        C1.append(C)  
    W = updateweight(U1, len(U1))
    #print("+++++")
    #print("+++++")
    #print(W)
    #print("+++++")
    #print("+++++")
    C = C1[0]
    for j in range(1, len(C1)):
        C = C+C1[j]
    U = initializeMembershipMatrix(C)
    center = calculateClusterCenter(U, W, C)
    #print(C)
    print("+++++")
    #print(U)
    C, U, labels = WFCM2(C, W, U, center)
    return C,labels
a=time.time()
for i in range(1,2):  
    max_iter = 20
    k=3
    # Number of data points
    n = len(df)
    m = 1.80
    sm=0
    ent=0
    pc=0
    vpp=0
    g=0
    p=k
    x.append(k)
    Dm=0
    myv=0
    calentropy=0
    C,labels = oFCM(p)

    print("$$")
b=time.time()
print("time is",b-a);
#print(C)
#print (len(df))
