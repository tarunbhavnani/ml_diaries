# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:35:33 2024

@author: tarun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
plt.style.use('ggplot')
import os
os.chdir(r"D:\kaggle\santa-2023")


info= pd.read_csv(r"D:\kaggle\santa-2023\puzzle_info.csv") 
puzzles= pd.read_csv(r"D:\kaggle\santa-2023\puzzles.csv")
ss=pd.read_csv(r"D:\kaggle\santa-2023\sample_submission.csv")

# =============================================================================


info.head()


[(i,len(set(puzzles[i]))) for i in puzzles]


puzzles.groupby("puzzle_type")['id'].count().sort_values(ascending=False).reset_index()


puzzles.groupby("puzzle_type")['num_wildcards'].value_counts().unstack().fillna(0).astype(int).sort_values(by=0, ascending=False).style.background_gradient()


info.iloc[0].allowed_moves


sol_state=puzzles.iloc[0]['solution_state'].split(';')
ini_state= puzzles.iloc[0]['initial_state'].split(';')
allowed= info.iloc[0]['allowed_moves']
allowed= ast.literal_eval(allowed)


# new={}
# for i in allowed:
#     #print(i)
#     new["-"+i]= list(np.argsort(allowed[i]))
    
# allowed.update(new)


#info['all_moves']=info['allowed_moves']
pt=[]
all_moves=[]
for i in range(0,len(info)):
    print(i)
    allowed= info.iloc[i]['allowed_moves']
    allowed= ast.literal_eval(allowed)
    new={}
    for j in allowed:
        new["-"+j]= list(np.argsort(allowed[j]))
    
    allowed.update(new)
    all_moves.append(allowed)
    pt.append(info.iloc[i]['puzzle_type'])
    #print(allowed)
    #info['all_moves'].iloc[i]= allowed

info_updated= pd.DataFrame({"puzzle_type":pt,"allowed_moves":all_moves })


# =============================================================================

#lets do it for the first sol

puzzles.iloc[0]

ptype= puzzles.iloc[0]['puzzle_type']

moves= info_updated[info_updated['puzzle_type']==ptype]['allowed_moves'][0]

[i for i in moves]

ini=puzzles.iloc[0]['initial_state']
ini=ini.split(";")

#check if r- and -ro , how they work

ss1=[ini[i] for i in moves['r0']]
ss2=[ss1[i] for i in moves['-r0']]
ss2==ini
#they are truely inverse, that means one after the other makes no diff


#function to remove when inverse follows
import re
# remove when inverse follows
def remove_inv(sel):
    ret=1
    for i in range(0,len(sel)-1):
        if re.sub("-", "",sel[i] )==re.sub("-", "",sel[i+1] ):
            ret=0
    return ret


#lets find all the combinations of all the moves from above

#mooves can be in groups of 1,2,3,...n, where n is the number of all moves
from itertools import combinations,permutations

diff_moves=[i for i in moves]
#len(moves)
#i=1

all_comb_moves={}
for i in range(1, len(moves)+1):
    print(i)
    ways_to_select = list(permutations(diff_moves, i))
    
    #remove when + follows - or vice versa
    ways_to_select=[i for i in ways_to_select if remove_inv(i)==1]
    
    for move in ways_to_select:
        #break
        for num,m in enumerate(move):
            if num==0:
                mv= moves[m]
            else:
                mv= [mv[i] for i in moves[m]]
        all_comb_moves[move]=mv


#write a function with ptype for this

def all_combinations_moves(ptype):
    moves= info_updated[info_updated['puzzle_type']==ptype]['allowed_moves'][0]
    diff_moves=[i for i in moves]
    all_comb_moves={}
    for i in range(1, len(moves)+1):
        print(i)
        ways_to_select = list(combinations(diff_moves, i))
        
        #remove when + follows - or vice versa
        ways_to_select=[i for i in ways_to_select if remove_inv(i)==1]
        
        for move in ways_to_select:
            #break
            for num,m in enumerate(move):
                if num==0:
                    mv= moves[m]
                else:
                    mv= [mv[i] for i in moves[m]]
            all_comb_moves[move]=mv
    return all_comb_moves
    
    
#problem
#check why this comb is not there-->'f1.d0.-r0.-f1.
[i for i in ways_to_select if len(i)>3 and i[0]=="f1" and i[1]=="d0" and i[2]=='-r0' and i[3]=='-f1']



# =============================================================================
# #noe we have all the possible moves

#can we do a distance check after each move weather we find the a better sol after one set of moves?
# =============================================================================

#first puzzle


puzzles.iloc[0]

ptype= puzzles.iloc[0]['puzzle_type']

moves= info_updated[info_updated['puzzle_type']==ptype]['allowed_moves'][0]


ini=puzzles.iloc[0]['initial_state']
ini=ini.split(";")
sol_state=puzzles.iloc[0]['solution_state']
sol_state=sol_state.split(';')

# lets applly all the moves and see after which one is the distnace between the next and the final sol is the least

  
def levenshtein_distance(ini, sol):
    word1= "".join([i for i in ini])
    word2="".join([i for i in sol])
    m, n = len(word1), len(word2)
    
    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution
    
    # The bottom-right cell contains the Levenshtein distance
    return dp[m][n]

diff=100000000
next_move=[]
for move_ in all_comb_moves:
    ini2=[ini[i] for i in all_comb_moves[move_]]
    dist=levenshtein_distance(ini2, sol_state)
    if dist<=diff:
        diff=dist
        next_move.append([move_, dist])

#get the lement with the lowest 2nd element, if more than one than the one with least number of elements in the first element
next_move=sorted(next_move, key=lambda x: (x[1], len(x[0])))[0]


    
# =============================================================================
# #lets check for 2nd sol
# =============================================================================

puzzles.iloc[1]

ptype= puzzles.iloc[1]['puzzle_type']

all_comb_moves=all_combinations_moves(ptype)

#moves= info_updated[info_updated['puzzle_type']==ptype]['allowed_moves'][0]


ini=puzzles.iloc[1]['initial_state']
ini=ini.split(";")
sol_state=puzzles.iloc[1]['solution_state']
sol_state=sol_state.split(';')


    
    
diff=100000000
next_moves=[]
for move_ in all_comb_moves:
    ini2=[ini[i] for i in all_comb_moves[move_]]
    dist=levenshtein_distance(ini2, sol_state)
    if dist<=diff:
        diff=dist
        next_moves.append([ini, move_, dist])

#get the lement with the lowest 2nd element, if more than one than the one with least number of elements in the first element

min_moves= min([i[2] for i in next_moves])

if min_moves==0:
    next_move=sorted(next_moves, key=lambda x: (x[2], len(x[0])))[0]
else:
    next_move=[i for i in next_moves if i[2]==min_moves]






next_moves=[]
diff=100000000

for nm in next_move:
    #break
#mooove ini to next move and repeat

    ini_= [ini[i] for i in all_comb_moves[nm[1]]]
    #ini= nm[0]

    for move_ in all_comb_moves:
        ini2=[ini_[i] for i in all_comb_moves[move_]]
        dist=levenshtein_distance(ini2, sol_state)
        if dist<=diff:
            diff=dist
            next_moves.append([ini_, move_, dist])

min_moves= min([i[2] for i in next_moves])

if min_moves==0:
    next_move=sorted(next_moves, key=lambda x: (x[2], len(x[0])))[0]
else:
    next_move=[i for i in next_moves if i[2]==min_moves]



next_move=sorted(next_move, key=lambda x: (x[1], len(x[0])))[0]

ini= [ini[i] for i in all_comb_moves[next_move[0]]]



diff=100000000
next_move=[]
for move_ in all_comb_moves:
    ini2=[ini[i] for i in all_comb_moves[move_]]
    dist=levenshtein_distance(ini2, sol_state)
    if dist<=diff:
        diff=dist
        next_move.append([move_, dist])

next_move=sorted(next_move, key=lambda x: (x[1], len(x[0])))[1]

ini= [ini[i] for i in all_comb_moves[next_move[0]]]


    





            
        



    
    
    
    
    
    
























