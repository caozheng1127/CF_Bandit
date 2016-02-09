import numpy as np
import random
class CFUCBArticleStruct:
	def __init__(self, context_dimension, latent_dimension, lambda_, init="zero"):
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension+latent_dimension

		self.A2 = lambda_*np.identity(n = self.latent_dimension)
		self.b2 = np.zeros(self.latent_dimension)
		self.A2Inv = np.linalg.inv(self.A2)

		self.count = 0		
		if (init=="random"):
			self.V = np.random.rand(self.d)
		else:
			self.V = np.zeros(self.d)

	def updateParameters(self, user, click):
		self.count += 1

		self.A2 += np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])
		self.b2 += user.U[self.context_dimension:]*(click - user.U[:self.context_dimension].dot(self.V[:self.context_dimension]))
		self.A2Inv  = np.linalg.inv(self.A2)

		self.V[self.context_dimension:] = np.dot(self.A2Inv, self.b2)


class CFUCBUserStruct:
	def __init__(self, context_dimension, latent_dimension, lambda_, init="zero"):
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension+latent_dimension

		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)

		self.count = 0

		if (init=="random"):
			self.U = np.random.rand(self.d)
		else:
			self.U = np.zeros(self.d)
		self.U = np.zeros(self.d)
	def updateParameters(self, article, click):
		self.count += 1

		self.A += np.outer(article.V,article.V)
		self.b += article.V*click
		self.AInv = np.linalg.inv(self.A)				

		self.U = np.dot(self.AInv, self.b)		
	def getTheta(self):
		return self.U
	
	def getA(self):
		return self.A

	def getProb(self, alpha, article):
		mean = np.dot(self.U, article.V)
		#var = np.sqrt(np.dot(np.dot(article.V, self.AInv),  article.V))
		#var2 = np.sqrt(np.dot(np.dot(self.U[self.context_dimension:], article.A2Inv),  self.U[self.context_dimension:]))
		pta = mean
		return pta
class CFEgreedyAlgorithm:
	def __init__(self, context_dimension, latent_dimension, alpha, lambda_, n, itemNum, init="zero", epsilon_init = 100):  # n is number of users
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension + latent_dimension
		self.epsilon_init = epsilon_init
		
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(CFUCBUserStruct(context_dimension, latent_dimension, lambda_ , init)) 
		self.articles = []
		for i in range(itemNum):
			self.articles.append(CFUCBArticleStruct(context_dimension, latent_dimension, lambda_ , init)) 

		self.alpha = alpha
		self.alpha2 = alpha

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True 
		self.CanEstimateW = False
		self.time = 1
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			self.articles[x.id].V[:self.context_dimension] = x.contextFeatureVector[:self.context_dimension]
			x_pta = self.users[userID].getProb(self.alpha, self.articles[x.id])
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		
		epsilon = self.get_epsilon()
		if random.random() > epsilon:
			return articlePicked
		else:
			return random.choice(pool_articles) 

	def get_epsilon(self):
		return min(self.epsilon_init/self.time, 1)
	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		# article = self.articles[articlePicked.id]
		# user = self.users[userID]

		#self.articles[articlePicked.id].A2 -= np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])*(user.count)
		self.users[userID].updateParameters(self.articles[articlePicked.id], click)
		#user = self.users[userID]
		#self.articles[articlePicked.id].A2 += np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])*(user.count-1)

		#self.users[userID].A -= np.outer(article.V, article.V)*(article.count)
		self.articles[articlePicked.id].updateParameters(self.users[userID], click)
		#article = self.articles[articlePicked.id]
		#self.users[userID].A += np.outer(article.V, article.V)*(article.count-1)

	def getCoTheta(self, userID):
		return self.users[userID].U


