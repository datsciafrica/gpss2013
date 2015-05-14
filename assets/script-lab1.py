"""
1. Getting stated
"""

import numpy as np
import pylab as pb
#import sys
import GPy

pb.ion()

##########################################

D = 1
var = 1.
theta = 0.2

k = GPy.kern.rbf(D,var,theta)

print k
k.plot()

##########################################

k = GPy.kern.rbf(D)
theta = np.asarray([0.2,0.5,1.,2.,4.])
for t in theta:
    k['.*lengthscale']=t
    k.plot()

pb.legend(theta)

##########################################

#k = GPy.kern.exponential(D)
#k = GPy.kern.Matern32(D)
#k = GPy.kern.Matern52(D)
#k = GPy.kern.Brownian(D)
#k = GPy.kern.linear(D)
#k = GPy.kern.bias(D)
#k = GPy.kern.periodic_Matern32(D)
#k = GPy.kern.rbfcos(D,frequencies=1,bandwidths=.1)
#k.plot()

kb = GPy.kern.Brownian(input_dim=1)
kb.plot(x = 2,plot_limits=[0,5])
kb.plot(x = 4,plot_limits=[0,5],ls='--',color='r')
pb.ylim([-0.1,5.1])

##########################################

k = GPy.kern.Matern52(input_dim=2)
X = np.random.rand(50,2)
C = k.K(X,X)
np.linalg.eigvals(C)

"""
2. Sample paths form a GP
"""

k = GPy.kern.rbf(input_dim=1,lengthscale=0.2)

X = np.linspace(0.,1.,500)
X = X[:,None]                # reshape X to make it nxD

mu = np.zeros((500))         # vector of the means
C = k.K(X,X)                 # covariance matrix

# Generate 20 sample path
Z = np.random.multivariate_normal(mu,C,20)

pb.figure()
for i in range(20):
    pb.plot(X[:],Z[i,:])

"""
3. GP regression model
"""

X = np.linspace(0.05,0.95,10)[:,None]
Y = -np.cos(np.pi*X) +np.sin(4*np.pi*X) + np.random.randn(10,1)*0.2
pb.figure()
pb.plot(X,Y,'kx',mew=1.5)

##########################################

k = GPy.kern.rbf(input_dim=1, variance=1., lengthscale=.2)
m = GPy.models.GPRegression(X,Y,k)

#m = GPy.models.GPRegression(X,Y,k)
#m['rbf_var'] = 1.
#m['rbf_len'] = .2
#m['noise'] = 0.0025

m.constrain_positive('')
m.constrain_fixed('noise',0)

##########################################

m.optimize()
m.plot()

##########################################

m.constrain_positive('noise')
m.randomize()
m['noise'] = .01
m.optimize()
m.plot()

##########################################
Xp = np.linspace(0,1,500)[:,None]
Yp, Vp, up95, lo95 = m.predict(Xp,full_cov=True)

"""
4.  Uncertainty propagation
"""

def branin(X):
    y = (X[:,1]-5.1/(4*np.pi**2)*X[:,0]**2+5*X[:,0]/np.pi-6)**2+10*(1-1/(8*np.pi))*np.cos(X[:,0])+10
    return(y[:,None])

xg1 = np.linspace(-5,10,5)
xg2 = np.linspace(0,15,5)

X = np.zeros((xg1.size * xg2.size,2))
for i,x1 in enumerate(xg1):
    for j,x2 in enumerate(xg2):
        X[i+xg1.size*j,:] = [x1,x2]

Y = branin(X)

#########################################

kg = GPy.kern.rbf(D=2, ARD = True)  # the flag ARD allows to set on lengthscale per dimension
kb = GPy.kern.bias(D=2)

k = kg + kb
k.plot()

m = GPy.models.GP_regression(X,Y,k,normalize_Y=True)
m.constrain_bounded('rbf_var',1e-3,1e5)
m.constrain_bounded('bias_var',1e-3,1e5)

m.constrain_bounded('rbf_len',.1,200.)
m.constrain_fixed('noise',1e-5)

m.randomize()
m.optimize()

m.plot()

# Compute the mean of model prediction on 1e5 Monte Carlo samples
Xp = np.random.uniform(size=(1e5,2))
Xp[:,0] = Xp[:,0]*15-5
Xp[:,1] = Xp[:,1]*15
Yp = m.predict(Xp)[0]
np.mean(Yp)


##### MC proba
#YM = []
#for i in range(50):
#    print i
#    Xmc = np.random.uniform(size=(1e7,2))
#    Xmc[:,0] = Xmc[:,0]*15-5
#    Xmc[:,1] = Xmc[:,1]*15
#    YM += [np.mean(branin(Xmc)>200)]
#
#np.mean(YM)



