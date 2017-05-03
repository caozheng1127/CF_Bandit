from collections import Counter
from math import log
import numpy as np 
from random import *
from custom_errors import FileExists 

def gaussianFeature(dimension, argv):
	mean = argv['mean'] if 'mean' in argv else 0
	std = argv['std'] if 'std' in argv else 1

	mean_vector = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(np.zeros(dimension), stdev)

	l2_norm = np.linalg.norm(vector, ord = 2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		"This makes it uniform over the circular range"
		vector = (vector / l2_norm)
		vector = vector * (random())
		vector = vector * argv['l2_limit']

	if mean is not 0:
		vector = vector + mean_vector

	vectorNormalized = []
	for i in range(len(vector)):
		vectorNormalized.append(vector[i]/sum(vector))
	return vectorNormalized
	#return vector

def featureUniform(dimension, argv = None):
	vector = np.array([random() for _ in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm
	return vector

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def checkFileExists(filename):
	try:
		with open(filename, 'r'):
			return 1
	except IOError:
		return 0 

def fileOverWriteWarning(filename, force):
	if checkFileExists(filename):
		if force == True:
			print "Warning : fileOverWriteWarning %s"%(filename)
		else:
			raise FileExists(filename)


def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V
	# return np.reshape(M, M.shape[0]*M.shape[1])

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W
	# To-do: use numpy built-in function reshape.
	# return np.reshape(V, (C_dimension, int(len(V)/C_dimension)))

def fun(x, y, theta):
	obj = (1/2.0)*(np.dot(np.transpose(x), theta) - click)**2
	regularization = 0
	return obj + regularization

def evaluateGradient(x,y,theta, lambda_, regu ):
	if regu == 'l1':
		grad = x*(np.dot(np.transpose(x),theta) - y) + lambda_*np.sign(theta)  #Lasso 
	elif regu == 'l2':
		grad = x*(np.dot(np.transpose(x),theta) - y) + lambda_*theta           # Ridge                      
	return grad

def getcons(dim):
	cons = []
	cons.append({'type': 'eq','fun': lambda x : np.sum(x)-1})

	for i in range(dim):
		cons.append({'type' : 'ineq','fun' : lambda  x: x[i] })
		cons.append({'type' : 'ineq','fun' : lambda x: 1-x[i]})
	
	return tuple(cons)
def getbounds(dim):
	bnds = []
	for i in range(dim):
		bnds.append((0,1))
	return tuple(bnds)


class observation_entry():
	def __init__(self, user, articlePool, OptimalReward, noise):
		self.user = user
		self.articlePool = articlePool
		self.OptimalReward = OptimalReward
		self.noise = noise


