# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 2018

@author: suraj prakash patil
@venue: department of technology, sppu
@topic: draft2- LS-SVM-Regression-From-Scratch- basic example
"""

import numpy as np


########### Training ##################

#  training_vec=np.matrix([(1,3),(2,1),(0,1)])
#training_vec=np.matrix([5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6])
training_vec=np.array(range(1,500))

### types of class is 1 & -1
#training_label=np.matrix([10,12,10,12,10,12,10,12,10,12,10,12,10,12,10,12])
training_label= np.multiply(training_vec,2)

n=np.size(training_label)
c= 0.5 * (n/1)

box=[]
box.append(1/c)
box1=np.identity(n) * (box)

b=np.transpose(training_vec)
K= b*training_vec
del b

K=K+box1

b=np.transpose(training_label)
a= training_label*b
 
H= np.multiply(a,K)
H=K

A=np.zeros([n+1,n+1])

A[0,1:n+1]=training_label
A[1:n+1,1:n+1]=H
A[1:n+1,0]=training_label
A1=A
baux=np.zeros([n+1,1])
baux[1:n+1,0]=np.ones([1,n])

supp_vecs= training_vec

ainv= np.linalg.inv(A)
A=np.transpose(baux); B=ainv; 

x = [[sum(a * b for a, b in zip(A_row, B_col)) 
                        for B_col in zip(*B)]
                                for A_row in A]


del ainv,B,baux,H,K,box1

bias=x[0][0]

#alpha=np.multiply(b  , np.transpose( x[0][1:(n+1)]))
alpha=np.multiply(np.transpose(b), np.transpose( x[0][1:(n+1)]))


########### Testing ############

# test_sample=[(0.5,1.1)]
#test_sample=[(50,10,500)]
#test_sample=[np.array(range(410,550))]
test_sample=[np.array([11,4,550])]
testo= np.multiply(test_sample,2)

output=[]
len_test=len(test_sample[0][:])
for i in range(0,len_test):
    z1=np.transpose(np.multiply(supp_vecs  ,  test_sample[0][i]))
    F1= ( np.transpose(z1) * np.transpose( alpha)  )  +  bias
    o1=F1[0]/(n/2)
    output.append(o1)
    del o1

print('deired-output :')
print(testo[0][:])
print('prediction :')
print(output)
err= ((testo[0][:]- np.array(output)))
mse= np.mean(err*err/len_test)
print('mse :')
print(mse)








