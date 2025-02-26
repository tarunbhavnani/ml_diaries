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





# =============================================================================
# Understanding Maximum Likelihood Estimation for Gaussian Distributions

# =============================================================================



First, carefully examine the final data points. We need to decide which type of distribution most closely represents the data.

Let's assume it's Gaussian (normal distribution). The Gaussian distribution is characterized by two parameters: the mean (μ) and the standard deviation (σ).

Maximum Likelihood Estimation (MLE) is a method used to find the values of μ and σ that maximize the likelihood of the observed data, effectively resulting in the best-fitting curve for the data.

To do this, we need to calculate the total probability of observing all the data points, which is the joint probability distribution of all observed data points. This typically involves calculating some conditional probabilities, which can be complex. Therefore, we make an important assumption: each data point is generated independently of the others. This assumption simplifies the mathematics significantly. If the events (i.e., the process that generates the data) are independent, then the total probability of observing all the data is the product of the probabilities of observing each data point individually (i.e., the product of the marginal probabilities).

First, we find the probability density of observing a single data point given by the formula:

𝑃(𝑥;𝜇,𝜎)=1𝜎2𝜋exp(−(𝑥−𝜇)22𝜎2)P(x;μ,σ)= σ 2π​ 1​ exp(− 2σ 2 (x−μ) 2 ​ )

In this notation, the semicolon in 𝑃(𝑥;𝜇,𝜎)
P(x;μ,σ) indicates that μ and σ are parameters of the probability distribution. This should not be confused with conditional probability, which is typically represented with a vertical line, e.g., 
𝑃(𝐴∣𝐵)P(A∣B).

Suppose we have three data points: 9, 9.5, and 11. We would input these values into the probability density function, find the individual probabilities, and then multiply them to get the joint probability.

Next, we need to determine the values of μ and σ that maximize this joint probability.

To achieve this, we use calculus: we differentiate the likelihood function with respect to μ and σ, set the derivatives to zero, and solve for μ and σ. However, this process can be complex.

To simplify, we use the log-likelihood. Since the logarithm is a monotonically increasing function, maximizing the likelihood is equivalent to maximizing the log-likelihood.

Taking the logarithm of the likelihood function makes differentiation easier. We then proceed with partial differentiation:

Partially differentiate the log-likelihood with respect to μ, set it to zero, and solve for μ.
Partially differentiate the log-likelihood with respect to σ, set it to zero, and solve for σ.
By solving these equations, we obtain the MLE estimates for μ and σ.
                                        

