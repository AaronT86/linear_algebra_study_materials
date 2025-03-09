import numpy as np
import polars as pl

#simple linear regression
#must add vector of 1s so Beta0 can be calculated
x = np.matrix([[1,1],
               [1,2],
               [1,3],
               [1,4]])

print(x)

y = np.matrix([[1],
               [3],
               [3],
               [5]])

print(y)

print(np.linalg.det(np.matmul(np.transpose(x), x)))

inv_xtx = np.linalg.inv(np.matmul(np.transpose(x), x))
xty = np.matmul(np.transpose(x), y)
print(xty)

beta1, beta2 = np.matmul(inv_xtx, xty)
print(beta1, beta2)

#mlr example

x = np.matrix([[1,1,12],
               [1,2,15],
               [1,3,20],
               [1,4,22]])

print(x)

y = np.matrix([[1],
               [3],
               [3],
               [5]])

print(y)

print(np.linalg.det(np.matmul(np.transpose(x), x)))

inv_xtx = np.linalg.inv(np.matmul(np.transpose(x), x))
xty = np.matmul(np.transpose(x), y)
print(xty)

beta1, beta2, beta3 = np.matmul(inv_xtx, xty)
print(beta1, beta2, beta3)