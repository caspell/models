import numpy as np


'''
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print ( w*x )
print ( np.sum(w*x) )
print ( np.sum(w*x) + b )
'''

def AND ( x1, x2 ) :
    x = np.array([ x1, x2 ])
    w = np.array([ 0.5 , 0.5 ])
    b = -0.7
    r = np.sum((x * w)) + b
    if r <= 0 :
        return 0
    else :
        return 1

def NAND ( x1, x2 ) :
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    r = np.sum((x * w)) + b
    if r <= 0:
        return 0
    else:
        return 1

def OR ( x1, x2 ) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    r = np.sum((x * w)) + b
    if r <= 0:
        return 0
    else:
        return 1

def XOR ( x1, x2 ) :
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

print ('-------------')
print (AND( 1, 1 ))
print (AND( 0, 1 ))
print (AND( 1, 0 ))
print (AND( 0, 0 ))
print ('-------------')
print (NAND( 1, 1 ))
print (NAND( 0, 1 ))
print (NAND( 1, 0 ))
print (NAND( 0, 0 ))
print ('-------------')
print (OR( 1, 1 ))
print (OR( 0, 1 ))
print (OR( 1, 0 ))
print (OR( 0, 0 ))
print ('-------------')
print (XOR( 1, 1 ))
print (XOR( 0, 1 ))
print (XOR( 1, 0 ))
print (XOR( 0, 0 ))