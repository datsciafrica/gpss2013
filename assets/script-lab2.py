"""
1. Getting stated
"""
import numpy as np
import pylab as pb
#import sys
import GPy

#pb.ion()

X = np.sort(np.random.rand(50,1)*12)
k = GPy.kern.rbf(1)
K = k.K(X)
K += np.eye(50)*0.01
y = np.random.multivariate_normal(np.zeros(50),K).reshape(50,1)

##########################################

m = GPy.models.GPRegression(X,y)
m.ensure_default_constraints()
m.optimize()
m.plot()
mu,var = m._raw_predict(X)
pb.vlines(X[:,0], (mu-2*np.sqrt(var))[:,0], (mu+2*np.sqrt(var))[:,0],"r")

"""
2. Build a sparse model
"""

Z = np.random.rand(3,1)*12
m = GPy.models.SparseGPRegression(X,y,Z=Z)
print m

##########################################

mu,var = m._raw_predict(Z)
pb.vlines(Z[:,0], (mu-2*np.sqrt(var))[:,0], (mu+2*np.sqrt(var))[:,0],"r")


# Questions
#m.optimize()
#mu,var = m._raw_predict(m.Z)
#m.plot()
#pb.vlines(m.Z[:,0], (mu-2*np.sqrt(var))[:,0], (mu+2*np.sqrt(var))[:,0],"r")

"""
3. Classification
"""

X = np.random.rand(100,2)
y = np.where(X[:,1:]>.5*(np.sin(X[:,:1]*4*np.pi)+.5),1,0)
X[:,1:] += y*0.1
m = GPy.models.GPClassification(X,y)
m.randomize()
m.update_likelihood_approximation()
m.plot()

m.optimize()

#Questions
#m.plot()
#for i in range(10):
#    m.update_likelihood_approximation()
#    m.optimize()
#
#m.update_likelihood_approximation()
#m.plot()

"""
4. Sparse GP classification

"""
import urllib
from scipy import io

urllib.urlretrieve('http://www.cs.nyu.edu/~roweis/data/olivettifaces.mat',\
'faces.mat')
face_data = io.loadmat('faces.mat')

##########################################

faces = face_data['faces'].T
pb.imshow(faces[120].reshape(64,64,order='F'),
    interpolation='nearest',cmap=pb.cm.gray)

urllib.urlretrieve(\'http://staffwww.dcs.sheffield.ac.uk/people/J.Hensman/gpsummer/datasets/has_glasses.np','has_glasses.np')

y = np.load('has_glasses.np')
y = np.where(y=='y',1,0).reshape(-1,1)

##########################################

index = np.random.permutation(faces.shape[0])
num_training = 200
Xtrain = faces[index[:num_training],:]
Xtest = faces[index[num_training:],:]
ytrain = y[index[:num_training],:]
ytest = y[index[num_training:]]

##########################################

from scipy import cluster
M = 8
Z, distortion = cluster.vq.kmeans(Xtrain,M)

##########################################

k = GPy.kern.rbf(4096,lengthscale=50) + GPy.kern.white(4096,0.001)
m = GPy.models.SparseGPClassification(Xtrain, ytrain, kernel=k, Z=Z, normalize_X=True)
m.update_likelihood_approximation()
m.ensure_default_constraints()

##########################################

pb.figure()
pb.imshow(m.dL_dZ()[0].reshape(64,64,order='F'),interpolation='nearest',cmap=pb.cm.gray)


#Questions
#for i in range(20):
#    m.optimize(max_f_eval=25)
#    m.update_likelihood_approximation()
#pb.imshow(m.dL_dZ()[0].reshape(64,64,order='F'),interpolation='nearest',cmap=pb.cm.gray)



