#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:33:07 2019

@author: tarun.bhavnani
"""

#manual nn


inpu= [.05,.1]
outpu=[.01,.99]

one hidden layer one output layer

2 neuron in both

#hidden layer neuron 1

net1= w1*.05+w2*.1+b1*1                          #----------------------w1
net1= .15*.05+.2*.1+.35*1

import numpy as np
def sigmoid(n):
    m=1/(1+np.exp(-n))
    return(m)
    
outh1= sigmoid(net1)

net2= w3*.05+w4*.1+b1*1
net2= .25*.05+.3*.1+.35*1
outh2= sigmoid(net2)


#output layer
neto1=.4*outh1+.45*outh2+.6*1  #w5,w6,b2----------------------line w5

outo1=sigmoid(neto1)

neto2=.5*outh1+.55*outh2+.6*1  #w7,w8,b2
outo2= sigmoid(neto2)


#calculate the error!
#squared error function
error total= sigma(1/2(target-output)^2)

e1=outpu[0]-outo1
e2=outpu[1]-outo2
et=.5*(outpu[0]-outo1)**2+.5*(outpu[1]-outo2)**2
et= .5*(e1)**2+.5*(e2)**2#----------------line et

#frrward pass complete

#backword pass

"each weight updates one by one"

#lets see for w5, i.e .4

partial derrivative(pd1) of et by w5:
#write d(et)/d() * d()/d() * d()/d(w5) and fill from behind
d(et)/d(w5)--> d(et)/ d(outo1) * d(outo1)/d(neto1) * d(neto1)/d(w5)

see the middle d(outo1)/d(neto1)
outo1 is the sigmoid of net01
derrivative of sigmoid is out(1-out)

d(et)/d(outo1)--> 2*.5*(outpu[0]-outo1)*(-1) #see line et

d(outo1)/d(neto1)--> outo1*(1-outo1)

d(neto1)/d(w5)--> outh1    # see line w5

pd1=2*.5*(outpu[0]-outo1)*(-1) * outo1*(1-outo1)  * outh1  

#similarly
d(et)/d(outo2) -->2*.5*(outpu[1]-outo2)*(-1)


#learning rate is .5

lr=.5
#w5=.4
w5 = w5-lr*pd1


similarly we find w6, w7, w8,b2



#now updating w1,w2,w3,w4 which are in the hidden layer
#it is a little more tricky

#doing like before

d(et)/d(w1)= d(et)/d() * d()/d() * d()/d(w1)


d(et)/d(outh1) * d(outh1)/d(net1) * d(net1)/d(w1)# 1,2,3


#1-->
#but both the outputs affect here so 
et= e1+e2 # see line et
d(et)/d(outh1)--> d(e1)/d(outh1) + d(e2)/d(outh1)


d(e1)/d(outh1)--> d(e1)/d(outo1)  *  d(outo1)/d(dneto1) * d(neto1)/d(outh1)

#d(et)/d(outo1)--> 2*.5*(outpu[0]-outo1)*(-1) from line et

2*.5*(outpu[0]-outo1)*(-1)*(outo1*(1-outo1))*.4

#-.055

d(e2)/d(outh1)--> d(e2)/d(outo2)  *  d(outo2)/d(dneto2) * d(neto2)/d(outh1)
#same as above

2*.5*(outpu[1]-outo2)*(-1)*(outo2*(1-outo2))*.5

#-0.019049118258278114

#adding both 
d(et)/d(outh1)-->

one=2*.5*(outpu[0]-outo1)*(-1)*(outo1*(1-outo1))*.4+2*.5*(outpu[1]-outo2)*(-1)*(outo2*(1-outo2))*.5
#03635030639314468

#this is #1

#2--> 
two=outh1*(1-outh1)

#3-->
three=.05


pd2=one*two*three

#0.00043856773447434685

#w1=.15
w1= w1- lr*pd2

#similarly update w2,w3,w4,b1















