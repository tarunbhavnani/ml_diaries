#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:23:03 2018

@author:https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
"""

a=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
a.shape
wh=np.array([[.42,.88,.55],[.1,.73,.68],[.60,.18,.47],[.92,.11,.52]])
wh.shape
bh=np.array([.46,.72,.08])
bh.shape

h=a.dot(wh)+bh
h.shape

def sigmoid(n):
    m=1/(1+np.exp(-n))
    return(m)

#wh2=np.array([[.81,.86,.75],[.92, .87,.83],[.81,.83,.78]])
wh2=sigmoid(h)
wh2.shape

bh2=np.array([])

wout=np.array([[.3],[.25],[.23]])

bout=np.array([.69])

out= sigmoid(wh2.dot(wout)+bout)

real=np.array([[1],[1],[0]])

E=real-out

#derrivative of sigmoid=x * (1 - x)

#slope_output_layer= derivatives_sigmoid(output)
Slope_output_layer=out*(1-out)

#Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
Slope_hidden_layer=wh2*(1-wh2)

lr=.1
d_output = E * Slope_output_layer

Error_at_hidden_layer = d_output.dot( wout.T)

d_hiddenlayer = Error_at_hidden_layer * Slope_hidden_layer


#update wt at both output and hidden layer

wout = wout + wh2.T.dot( d_output)*lr
wh =  wh+ a.T.dot(d_hiddenlayer)*lr

#Step 11: Update biases at both output and hidden layer

bh.shape
bh=bh.reshape(1,3)
bh += np.sum(d_hiddenlayer, axis=0,keepdims=True)

bout.shape
bout=bout.reshape(1,1)
bout += np.sum(d_output, axis=0,keepdims=True) *lr


