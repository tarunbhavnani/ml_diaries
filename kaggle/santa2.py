# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:18:22 2024

@author: tarun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from itertools import combinations
import re
plt.style.use('ggplot')
import os
os.chdir(r"D:\kaggle\santa-2023")


info= pd.read_csv(r"D:\kaggle\santa-2023\puzzle_info.csv") 
puzzles= pd.read_csv(r"D:\kaggle\santa-2023\puzzles.csv")
ss=pd.read_csv(r"D:\kaggle\santa-2023\sample_submission.csv")

# =============================================================================
# add all inverse moves
# =============================================================================

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
# club all possible combinations of moves 
# =============================================================================

# remove when inverse follows
def remove_inv(sel):
    ret=1
    for i in range(0,len(sel)-1):
        if re.sub("-", "",sel[i] )==re.sub("-", "",sel[i+1] ):
            ret=0
    return ret

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





# =============================================================================
# levenshetein dist
# =============================================================================

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

# =============================================================================
# BFS
# =============================================================================



# =============================================================================
# add all inverse moves
# =============================================================================

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
# 
# =============================================================================
from collections import deque

def apply_move(state, move):
    return [state[i] for i in move]

def apply_move_np(state, move):
    return np.array(state)[move].tolist()

def find_solution(initial, final, moves):
    visited = set()
    queue = deque([(initial, [])])

    while queue:
        current_state, current_moves = queue.popleft()
        if current_state == final:
            return current_moves
        prev=""
        for move_name, move in moves.items():
            
            print(move_name)
            #can we remove the ones that dont matter, yes for inverse 
            if re.sub("-","",prev)==re.sub("-","",move_name) and prev!=move_name:
                prev=move_name
                pass
            else:
                prev=move_name
            
                #new_state = apply_move(current_state, move)
                new_state = apply_move_np(current_state, move)
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    queue.append((new_state, current_moves + [move_name]))

    return None




# =============================================================================
# 
# =============================================================================
import re

list(puzzles)

ptype="cube_2/2/2"
#ptype="wreath_100/100"
initial_state=puzzles[puzzles['puzzle_type']==ptype]['initial_state'].iloc[0]
initial_state=str(initial_state).split(';')
solution_state=puzzles[puzzles['puzzle_type']==ptype]['solution_state'].iloc[0]
solution_state=solution_state.split(';')

moves= info_updated[info_updated['puzzle_type']==ptype]['allowed_moves'].iloc[0]

%%time
solution = find_solution(initial_state, solution_state, moves)
#iloc[1] for cube 222
#['f0', 'r1', 'f1', 'd0', 'd0', 'f1', 'd1', '-r1', '-d1']
#13 mins
apply_move_np(initial_state,moves['f0'])

# =============================================================================
# lets do same with networkx
# =============================================================================



import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to represent your puzzle or problem
# For example, adding edges based on moves
for move_name, move in moves.items():
    G.add_edge(tuple(initial_state), tuple(apply_move(initial_state, move)))



# Perform BFS
start_node = tuple(initial_state)
end_node = tuple(solution_state)
path = nx.shortest_path(G, source=start_node, target=end_node)

# Print the sequence of moves
if path:
    print("Sequence of moves:", [move_name for i, move_name in enumerate(path) if i < len(path) - 1])
else:
    print("No solution found.")

# Optionally, you can visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()
















