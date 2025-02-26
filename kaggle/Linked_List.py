# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:19:46 2024

@author: tarun
"""


# =============================================================================
# linked list
# =============================================================================

# Definition for singly-linked list node
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Function to traverse a linked list
def traverse_linked_list(head):
    current = head  # Start at the head node
    while current:
        # Access the value of the current node
        print(current.val, end=" -> ")
        # Move to the next node
        current = current.next
    print("None")  # Print None to indicate the end of the list

# Example usage
# Create a linked list: 1 -> 2 -> 3 -> None

def create_ll(values):
    head= ListNode()
    current=head
    for value in values:
        current.next= ListNode(value)
        current=current.next
    
    return head.next

l1= create_ll([2,4,3])
l2= create_ll([5,6,4])


#You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

#You may assume the two numbers do not contain any leading zero, except the number 0 itself.


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        #l3= sum([10**i*j for i,j in enumerate(l1)])+sum([10**i*j for i,j in enumerate(l2)])
        
        #return [int(i) for i in str(l3)][::-1]
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        
        l1_head= l1
        l2_head=l2
        counter=0
        new=0
        while l1_head or l2_head:
            try:
                new+= 10**counter*(l1_head.val)
                l1_head= l1_head.next
            except:
                pass
            try:
                new+= 10**counter*(l2_head.val)
                l2_head= l2_head.next
            except:
                pass
            counter+=1
            
        #new3=new+new1
        head= ListNode()
        current=head
        for i in [int(i) for i in str(new)][::-1]:
            current.next=ListNode(i)
            current=current.next
        return head.next
            

