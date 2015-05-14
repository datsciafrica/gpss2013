import numpy as np
import pylab as pb

#olympics = scipy.io.loadmat('olympics.mat')
#import scipy.io
#olympics.keys()

pb.ion()                     # Open one thread per plot

# Download this data set: http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/olympicMarathonTimes.csv

olympics = np.genfromtxt('olympicMarathonTimes.csv', delimiter=',')


"""
1. Linear regression: iterative solution
"""

x = olympics[:, 0:1]
y = olympics[:, 1:2]
print(x)
print(y)

pb.plot(x, y, 'rx')

##########################################

N = y.size
m = 0
c = np.sum(y-m*x)/N
sigma2 = 1.

def error(m,c,sigma2):
    return .5*N * np.log(sigma2) + .5*np.sum((y-m*x-c)**2.)/sigma2

err0 = error(m,c,sigma2)

delta = np.inf
while delta > 1e-10:
    c = np.sum(y-m*x)/N
    m = np.sum(x*(y-c))/np.sum(x**2)
    err = error(m,c,sigma2)
    delta = abs(err - err0)
    err0 = err

sigma2 = np.sum((y-m*x-c)**2)/N
xTest = np.linspace(1890,2010,100)
#pb.plot(xTest,m*xTest + c,"k-")

#t_hat = m*x + c
#e = t_hat - y
#print sigma2,e.var()

"""
2. Basis functions
"""
Phi = np.hstack((np.ones(x.shape), x))
print(Phi)

wStar = np.linalg.solve(np.dot(Phi.T,Phi),np.dot(Phi.T,y))
print wStar[1],m
print wStar[0],c

PhiTest = np.zeros((xTest.shape[0], 2))
PhiTest[:, 0] = np.ones(xTest.shape)
PhiTest[:, 1] = xTest

yTest = np.dot(PhiTest, wStar)
#pb.plot(xTest, yTest, 'b-')


##########################################
order = 3

Phi = np.zeros((x.shape[0], order))
for i in range(0, order):
    Phi[:, i] = x.T**i

wStar = np.linalg.solve(np.dot(Phi.T,Phi),np.dot(Phi.T,y))
PhiTest = np.zeros((xTest.shape[0], order))
for i in range(0, order):
    PhiTest[:, i] = xTest.T**i
yTest = np.dot(PhiTest, wStar)
#pb.plot(xTest, yTest, 'b-')
