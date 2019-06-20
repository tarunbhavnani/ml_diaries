#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:39:29 2019

@author: tarun.bhavnani


First thing is have a keen kook at the final data points.
WE have to decide what kind of distribution most closely represents the data.


lets say its gaussian.
Gaussian has two parameters std deviation and mean.

Maximum likelihood estimation is a method that will find the values of μ and σ that result
 in the curve that best fits the data.
 

What we want to calculate is the total probability of observing all of the data, i.e. 
the joint probability distribution of all observed data points. To do this we would
 need to calculate some conditional probabilities, which can get very difficult. So
 it is here that we’ll make our first assumption. The assumption is that each data
 point is generated independently of the others. This assumption makes the maths much 
 easier. If the events (i.e. the process that generates the data) are independent, 
 then the total probability of observing all of data is the product of observing each 
 data point individually (i.e. the product of the marginal probabilities).
first we will find the probability density of observing a single data point.

P(x; μ, σ)

P= (1/(sigma*(2*pi)**.5)) * np.exp(-((x-mu)**2)/(2*sigma**2))

The semi colon used in the notation P(x; μ, σ) is there to emphasise that the symbols 
that appear after it are parameters of the probability distribution. So it shouldn’t be 
confused with a conditional probability (which is typically represented with a vertical
 line e.g. P(A| B)).

Now we have three points 9,9.5 and 11

we will input inthe formula find the conditional probabilities and multiply them.


We just have to figure out the values of μ(mu) and σ(sigma) that results in giving the
 maximum value of the above expression.


How?
calculus
we diffrenciate and equate to zero.

but it gets pretty difficult here

ENTER log liklihood

since log is monotonically increasing thus we can use it

take log of bith sides and proceed with the differenciation.


Now we have to find the values for mu and sigma.
so we will do partial diffrenciation.
first we will partial diff with mu , equate to zero and find mu
then we will partial diff with sigma, equate to zero and find sigma.










                                        