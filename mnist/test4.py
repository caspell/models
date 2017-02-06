import numpy as np


values1 = np.loadtxt('test3.txt', unpack=True)

print ( values1 )

val1 = values1[:, 0]

val2 = values1[:, 1]





#print ( val1 )
print ( val2 )
values1[:, 1] = (val2 - val2.mean()) / val2.std()

print ( values1 )

#print ( ( val1 - val2 ).std() )


