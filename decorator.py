# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:27:47 2021

@author: ELECTROBOT
"""

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

def say_whee():
    print("Whee!")

say_whee = my_decorator(say_whee)

say_whee()

# =============================================================================


def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")

say_whee()


# =============================================================================
# 
# =============================================================================
def smart_divide(func):
    def inner(a, b):
        print("I am going to divide", a, "and", b)
        if b == 0:
            print("Whoops! cannot divide")
            return

        return func(a, b)
    return inner


@smart_divide
def divide(a, b):
    print(a/b)





def try_catch(func):
    def inner(*args):
        try:
            func(*args)
        except Exception as e:
            return str(e)
    return inner
        
@try_catch
def divide(a, b):
    print(a/b)


divide(2,3)
divide(2,0)
