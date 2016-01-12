import numpy as np
import random
import math
from sklearn.preprocessing import normalize
class EgreedyContextualStruct:
    def __init__(self, epsilon_init, userNum, itemNum,k, feature_dim, tau=0.1, lambda_=0.1, init='zero', learning_rate='decay'):
        self.reward = 0
        self.userNum = userNum
        self.itemNum = itemNum
        # self.R = np.zeros((userNum, itemNum))
        self.S = np.zeros((userNum, itemNum))
        self.time = 1
        self.tau = tau #SGD Learning rate
        self.tau_init = 1 # decay
        self.learning_rate = learning_rate
        self.epsilon_init = epsilon_init
        
        if (init == 'random'):
            self.U = np.random.rand(userNum,k)
            self.V = np.random.rand(itemNum,k)
        else:   
            self.U = np.zeros((userNum,k))         
            self.V = np.zeros((itemNum,k))
        
        self.lambda_ = lambda_ #
        self.feature_dim = feature_dim

        #add normalization
        self.U = normalize(self.U, axis=1, norm='l1')
        self.V = normalize(self.V, axis=1, norm='l1')
        self.k = k
        # print k

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False
    def decide(self, items, userID):
        max_r = float('-inf')
        max_itemID = None
        #print (items, userID)
        for item in items:
            itemID = item.id
            self.V[itemID, :self.feature_dim] = item.contextFeatureVector[:self.feature_dim]
            restimate = self.U[userID].dot(self.V[itemID])
            if (max_r < restimate):
                max_r = restimate
                max_itemID = item
        #print (max_r, max_itemID)
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            return max_itemID
        else:
            return random.choice(items) 
    def getPrediction(self, itemID, userID):
        return self.U[userID].dot(self.V[itemID, :])
    def getCoTheta(self, userID):
        return self.U[userID]
    def updateParameters(self, item, reward, userID):
        self.time += 1
        self.SGD(reward, item, userID)
        if (self.learning_rate=='decay'):
            self.tau = self.tau_init/math.sqrt(self.time/self.userNum+1)
    def get_epsilon(self):
        return min(self.epsilon_init/self.time, 1)

    def SGD(self, reward, item, userID):
        itemID = item.id
        self.V[itemID, :self.feature_dim] = item.contextFeatureVector[:self.feature_dim]
        restimate = self.U[userID].dot(self.V[itemID])        
        self.U[userID] += self.tau*((reward-restimate)*self.V[itemID]- self.lambda_*self.U[userID])
        self.V[itemID] += self.tau*((reward - self.U[userID, :self.feature_dim].dot(self.V[itemID, :self.feature_dim])-restimate)*self.U[userID]- self.lambda_*self.V[itemID])
        self.V[itemID, :self.feature_dim] = item.contextFeatureVector[:self.feature_dim]
        #print (self.U[userID], (reward-restimate)*self.V[itemID]- self.r*self.U[userID], (reward - self.U[userID, :self.feature_dim].dot(self.V[itemID, :self.feature_dim])-restimate)*self.U[userID][self.feature_dim:])
        # if userID == 1:
        #     print np.linalg.norm((reward-restimate)*self.V[itemID]- self.r*self.U[userID])#, (reward-restimate)*self.V[itemID]- self.r*self.U[userID]
    def getCoTheta(self, userID):
        return self.U[userID]
