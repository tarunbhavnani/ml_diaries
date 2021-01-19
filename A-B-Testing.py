#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:46:08 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#A/B testing

import random

N_A=random.choices(population=[1,0], weights=[.1,.9], k=1000)
N_B=random.choices(population=[1,0], weights=[.12,.88], k=1000)

df=pd.DataFrame(data= N_A)
df["group"]="A"

df1=pd.DataFrame(data= N_B)
df1["group"]="B"
list(df)

df= df.append(df1)

df.columns=["kl","group"]
list(df)

ab_summary = df.pivot_table(values='kl', index='group', aggfunc=np.sum)
# add additional columns to the pivot table
ab_summary['total'] = df.pivot_table(values='kl', index='group', aggfunc=lambda x: len(x))
ab_summary['rate'] = df.pivot_table(values='kl', index='group')

#data created!!

##
"Compare the Two Groups"

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,6))
x = np.linspace(A_converted-49, A_converted+50, 100)
y = scs.binom(A_total, A_cr).pmf(x)
ax.bar(x, y, alpha=0.5)
ax.axvline(x=B_cr * A_total, c='blue', alpha=0.75, linestyle='--')
plt.xlabel('converted')
plt.ylabel('probability')














