'''
Author: Akond Rahman 
Sep 09, 2022 
Code needed for Workshop 5 
'''

from ast import operator
import random 
import itertools


def simpleCalculator(v1, v2, operation):
    print('The operation is:', operation)
    valid_ops = ['+', '-', '*', '/']
    res = 0 
    if operation in valid_ops:
        if operation=='+':
            res = v1 + v2 
        elif operation=='-':
            res = v1 - v2 
        elif operation=='*':
            res = v1 * v2 
        elif operation=='/':                
            res = v1 / v2 
        elif operation=='%':                
            res = v1 % v2 
        print('After calculation the result is:' , res) 
    else: 
        print('Operation not allowed. Allowable operations are: +, -, *, /, %')
        print('No calculation was done.') 
    return res 


def checkNonPermissiveOperations(op_list): 
    for op_ in op_list:
        simpleCalculator( 100, 1, op_ )

def fuzzValues():
    output = []
    valid_chars = ''.join(map(chr, range(128)))
    vallists = itertools.combinations_with_replacement(valid_chars, 1) 
    for vallist in vallists:
        output.append( ''.join(vallist) )
    return output

def simpleFuzzer(): 
    # Complete the following methods 
    values = fuzzValues()
    checkNonPermissiveOperations(values) 


if __name__=='__main__':
    # val1, val2, op = 100, 1, '+'

    # data = simpleCalculator(val1, val2, op)
    # print('Operation:{}\nResult:{}'.format(  op, data  ) )

    simpleFuzzer()