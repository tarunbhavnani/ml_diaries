# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:37:04 2024

@author: tarun
"""

string= "ababababbabababa"

# def palin_check(string):
#     if len(string)%2==0:
#         return string[0:int(len(string)/2)]==string[int(len(string)/2):][::-1]
#     else:
        
#         return string[0:int(len(string)//2)]==string[int(len(string)//2)+1:][::-1]
        

# string= "tarunbhavnani"

# def longest_palin(string):
#     mx=0
#     ans=""
#     for j in range(len(string)+1):
#         st=string[j:]
#         for i in range(len(st)+1):
#             if palin_check(st[:i]):
#                 if len(st[:i])>mx:
#                     mx=len(st[:i])
#                     ans= st[:i]
                
#     return ans
        
# =============================================================================
# go through ech letter only onec and check if it is the middle of a palindrome
# =============================================================================

class Solution:
    def longestPalindrome(self, string: str) -> str:
        def expand_around_center(left, right):
            while left >= 0 and right < len(string) and string[left] == string[right]:
                left -= 1
                right += 1
            return string[left + 1:right]
        
        longest = ""
        for i in range(len(string)):
            # For odd length palindromes
            pal1 = expand_around_center(i, i)
            # For even length palindromes
            pal2 = expand_around_center(i, i + 1)
            longest = max(longest, pal1, pal2, key=len)
        return longest







    







