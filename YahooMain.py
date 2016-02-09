from conf import *     # it saves the address of data stored and where to save the data produced by algorithms
import argparse # For argument parsing
import time
import re             # regular expression library
from random import random, choice,shuffle     # for random strategy
from operator import itemgetter
import datetime
import numpy as np     
import sys
from scipy.sparse import csgraph
from scipy.spatial import distance
from YahooExp_util_functions import *


from CoLin import AsyCoLinUCBUserSharedStruct, AsyCoLinUCBAlgorithm, CoLinUCBUserSharedStruct
from GOBLin import GOBLinSharedStruct
from LinUCB import LinUCBUserStruct, Hybrid_LinUCBUserStruct
from CF_UCB import CFUCBAlgorithm
from CFEgreedy import CFEgreedyAlgorithm
from EgreedyContextual import EgreedyContextualStruct
from PTS import PTSAlgorithm

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.learn_stats = articleAccess()
class Article():    
    def __init__(self, aid, FV=None):
        self.id = aid
        self.featureVector = FV
        self.contextFeatureVector = FV

self.learn_stats = articleAccess()

if __name__ == '__main__':
    # regularly print stuff to see if everything is going alright.
    # this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
    def printWrite():
        randomLearnCTR = articles_random.learn_stats.updateCTR()
        recordedStats = [randomLearnCTR]
        for alg_name, alg in algorithms.items():
        	algCTR = alg.learn_stats.updateCTR()
            recordedStats.append(algCTR)
            recordedStats.append(alg.learn_stats.accesses)
            recordedStats.append(alg.learn_stats.clicks)        
        # write to file
        save_to_file(fileNameWrite, recordedStats, tim) 
    def WriteStat():
        with open(fileNameWriteStatTP, 'a+') as f:
            for key, val in articleTruePositve.items():
                f.write(str(key) + ';'+str(val) + ',')
            f.write('\n')
        with open(fileNameWriteStatTN, 'a+') as f:
            for key, val in articleTrueNegative.items():
                f.write(str(key) + ';'+str(val) + ',')
            f.write('\n')
        with open(fileNameWriteStatFP, 'a+') as f:
            for key, val in articleFalsePositive.items():
                f.write(str(key) + ';'+str(val) + ',')
            f.write('\n')


    def calculateStat():
        if click:        
            for article in currentArticles:
                if article == article_chosen:
                    articleTruePositve[article_chosen] +=1
                else:
                    articleTrueNegative[article] +=1                
        else:
            for article in currentArticles:
                if article == article_chosen:
                    articleFalsePositive[article_chosen] +=1

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--YahooDataFile', dest="Yahoo_save_address", help="input the adress for Yahoo data")
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, GOBLin, LinUCB, HybridLinUCB, CFUCB, CFEgreedy, SGDEgreedy, PTS.')

    parser.add_argument('--showheatmap', action='store_true',
                    help='Show heatmap of relation matrix.') 
    parser.add_argument('--userNum', dest = 'userNum', help = 'Set the userNum, can be 20, 40, 80, 160')

    parser.add_argument('--Sparsity', dest = 'SparsityLevel', help ='Set the SparsityLevel by choosing the top M most connected users, should be smaller than userNum, when equal to userNum, we are using a full connected graph')
    parser.add_argument('--diag', dest="DiagType", help="Specify the setting of diagional setting, can be set as 'Orgin' or 'Opt' ") 

    parser.add_argument('--particle_num', 
                        help='Particle number for PTS.')
    parser.add_argument('--dimension', 
                        help='Feature dimension used for estimation.')

    args = parser.parse_args()

    algName = str(args.alg)
    clusterNum = int(args.userNum)
    SparsityLevel = int(args.SparsityLevel)
    yahooData_address = str(args.Yahoo_save_address)
    DiagType = str(args.DiagType)
    
    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')     # the current data time
    dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    fileSig = str(algName)+str(clusterNum)+ 'SP'+ str(SparsityLevel)+algName
    batchSize = 2000
    statBatchSize = 200000                            # size of one batch
    
    d = 5             # feature dimension
    alpha = 0.3     # control how much to explore
    lambda_ = 0.2   # regularization used in matrix A
    epsilon = 0.3
    
    totalObservations = 0

    articleTruePositve = {}
    articleFalseNegative = {}

    articleTrueNegative = {}
    articleFalsePositive = {}

    fileNameWriteCluster = os.path.join(Kmeansdata_address, '10kmeans_model'+str(clusterNum)+ '.dat')
    userFeatureVectors = getClusters(fileNameWriteCluster)    
    userNum = clusterNum
    if DiagType == 'Orgin':
        W = initializeW(userFeatureVectors, SparsityLevel)
    elif DiagType == 'Opt':
        W = initializeW_opt(userFeatureVectors, SparsityLevel)   # Generate user relation matrix
    GW = initializeGW(W , epsilon)
     
    articles_random = randomStruct()
   	algorithms = {}
    runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB= run_CFUCB = run_CFEgreedy = run_SGDEgreedy = False
    if args.alg:
        if args.alg == 'CoLinUCB':
            runCoLinUCB = True
            algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = userNum, W = W)
        elif args.alg == 'GOBLin':
            runGOBLin = True
        elif args.alg == 'LinUCB':
            runLinUCB = True
            algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = OriginaluserNum)
        elif args.alg =='M_LinUCB':
            run_M_LinUCB = True
        elif args.alg == 'Uniform_LinUCB':
            run_Uniform_LinUCB = True
        elif args.alg == 'CFUCB':
            run_CFUCB = True
            algorithms['CFUCB'] = CFUCBAlgorithm(context_dimension = context_dimension, latent_dimension = latent_dimension, alpha = 0.2, alpha2 = 0.1, lambda_ = lambda_, n = OriginaluserNum, itemNum=itemNum, init='random')
        elif args.alg == 'CFEgreedy':
            run_CFEgreedy = True
            algorithms['CFEgreedy'] = CFEgreedyAlgorithm(context_dimension = context_dimension, latent_dimension = latent_dimension, alpha = 200, lambda_ = lambda_, n = OriginaluserNum, itemNum=itemNum, init='random')
        elif args.alg == 'SGDEgreedy':
            run_SGDEgreedy = True
            algorithms['SGDEgreedy'] = EgreedyContextualStruct(epsilon_init=200, userNum=OriginaluserNum, itemNum=itemNum, k=context_dimension+latent_dimension, feature_dim = context_dimension, lambda_ = lambda_, init='random', learning_rate='constant')

        elif args.alg == 'ALL':
            runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB=True
    else:
        args.alg = 'Random'

    AlgReward = {}
    AlgPicked = {}
    AlgRegret = {}
    for alg_name, alg in algorithms.items():
        AlgReward[alg_name] = []
        AlgPicked[alg_name] = []
        AlgRegret[alg_name] = []
        alg.reward = 0    
    for i in range(userNum):
        LinUCB_users.append(LinUCBStruct(d, lambda_ ))
    
    EgreedyContextual = EgreedyContextualStruct(Tu= 200, m=10, lambd=0.1, alpha=2000, userNum=userNum, itemNum=200000, k=2+5, feature_dim = 5, init='zero')
    for dataDay in dataDays:
        fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay    +'.'+ str(userNum) +'.userID'
        fileNameWrite = os.path.join(Yahoo_save_address, fileSig + dataDay + timeRun + '.csv')

        fileNameWriteStatTP = os.path.join(Yahoo_save_address, 'Stat_TP'+ fileSig + dataDay + timeRun + '.csv')
        fileNameWriteStatTN = os.path.join(Yahoo_save_address, 'Stat_TN'+ fileSig + dataDay + timeRun + '.csv')
        fileNameWriteStatFP = os.path.join(Yahoo_save_address, 'Stat_FP'+ fileSig + dataDay + timeRun + '.csv')

        # put some new data in file for readability
        with open(fileNameWrite, 'a+') as f:
            f.write('\nNewRunat  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
            f.write('\n,Time,RandomCTR;'+ str(algName) + 'CTR;' + 'accesses;'+ 'clicks;' + '' +'\n')

        print fileName, fileNameWrite
        with open(fileName, 'r') as f:
            # reading file line ie observations running one at a time
            for line in f:
                totalObservations +=1

                tim, article_chosen, click, currentUserID, pool_articles = parseLine_ID(line)
                #currentUser_featureVector = user_features[:-1]
                #currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                
                #-----------------------------Pick an article (CoLinUCB, LinUCB, Random)-------------------------
                articlePool = []                
                for article in pool_articles:
                    article_id = int(article[0])
                    article_featureVector =np.asarray(article[1:6])
                    articlePool.append(Article(article_id, article_featureVector))
                shuffle(articlePool)
                for alg_name, alg in algorithms.items():
	                pickedArticle = alg.decide(articlePool, userID)
	                # reward = getReward(userID, pickedArticle) 
	                if (pickedArticle.id == article_chosen):
	                    reward = 1
	                else:
	                    reward = 0
	                alg.updateParameters(pickedArticle, reward, userID)
	                alg.reward += reward
	                AlgReward[alg_name].append(reward)
	                AlgPicked[alg_name].append(pickedArticle.id)
	                regret = OptimalReward - reward 
	                AlgRegret[alg_name].append(regret) 
	                
                    # CoLinUCB pick article
                    if len(article_featureVector)==5:
                        currentArticles.append(article_id)            
                        if algName == 'CoLin':
                            CoLinUCB_pta = CoLinUCB_USERS.getProb(alpha, article_featureVector, currentUserID)
                            if CoLinUCB_maxPTA < CoLinUCB_pta:
                                CoLinUCBPicked = article_id    # article picked by CoLinUCB
                                CoLinUCB_PickedfeatureVector = article_featureVector
                                CoLinUCB_maxPTA = CoLinUCB_pta
                        if algName == 'GOBLin':
                            GOBLin_pta = GOBLin_USERS.getProb(alpha, article_featureVector, currentUserID)
                            if GOBLin_maxPTA < GOBLin_pta:
                                GOBLinPicked = article_id    # article picked by GOB.Lin
                                GOBLin_PickedfeatureVector = article_featureVector
                                GOBLin_maxPTA = GOBLin_pta
                        if algName == 'HybridLinUCB':
                            HybridLinUCB_pta = HybridLinUCB_USERS.getProb(alpha, article_featureVector, currentUserID)
                            if HybridLinUCB_maxPTA < HybridLinUCB_pta:
                                HybridLinUCBPicked = article_id
                                HybridLinUCB_PickedfeatureVector = article_featureVector
                                HybridLinUCB_maxPTA = HybridLinUCB_pta
                             
                        if algName == 'LinUCB':
                            LinUCB_pta = LinUCB_users[currentUserID].getProb(alpha, article_featureVector)
                            if LinUCB_maxPTA < LinUCB_pta:
                                LinUCBPicked = article_id    # article picked by CoLinU
                                LinUCB_PickedfeatureVector = article_featureVector
                                LinUCB_maxPTA = LinUCB_pta
                        if algName == 'EgreedyContextual':
                            EgreedyContextual_pta = EgreedyContextual.getProb(article_id, currentUserID, article_featureVector) 
                            if EgreedyContextual_maxPTA < EgreedyContextual_pta:
                                EgreedyContextualPicked = article_id
                                EgreedyContextual_PickedfeatureVector = article_featureVector
                                EgreedyContextual_maxPTA = EgreedyContextual_pta
                if algName == 'EgreedyContextual':    
                    if random() < EgreedyContextual.get_epsilon():
                        i = choice(range(len(currentArticles)))
                        EgreedyContextualPicked = currentArticles[i]
                        EgreedyContextual_PickedfeatureVector = np.asarray(pool_articles[i][1:6])

                for article in currentArticles:
                    if article not in articleTruePositve:
                        articleTruePositve[article] = 0
                        articleTrueNegative[article] = 0
                        articleFalsePositive[article] = 0
                        articleFalseNegative[article] = 0

                # article picked by random strategy
                articles_random.learn_stats.addrecord(click)
                if algName == 'CoLin':
                    if CoLinUCBPicked == article_chosen:
                        CoLinUCB_USERS.learn_stats.addrecord(click)
                        CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector, click, currentUserID)
                        calculateStat()

                if algName == 'GOBLin':
                    if GOBLinPicked == article_chosen:
                        GOBLin_USERS.learn_stats.addrecord(click)
                        GOBLin_USERS.updateParameters(GOBLin_PickedfeatureVector, click, currentUserID)
                        calculateStat()
                if algName == 'HybridLinUCB':
                    if HybridLinUCBPicked == article_chosen:
                        HybridLinUCB_USERS.learn_stats.addrecord(click)
                        HybridLinUCB_USERS.updateParameters(HybridLinUCB_PickedfeatureVector, click, currentUserID)
                        calculateStat()
                if algName == 'LinUCB':
                    #print 'Picked', LinUCBPicked, click,LinUCB_maxPTA, article_chosen
                    if LinUCBPicked == article_chosen:
                        LinUCB_users[currentUserID].learn_stats.addrecord(click)
                        LinUCB_users[currentUserID].updateParameters(LinUCB_PickedfeatureVector, click)
                        calculateStat()
                if algName == 'EgreedyContextual':
                    if EgreedyContextualPicked == article_chosen:
                        EgreedyContextual.learn_stats.addrecord(click)
                        EgreedyContextual.updateParameters(click, EgreedyContextualPicked, currentUserID, EgreedyContextual_PickedfeatureVector)
                        calculateStat()
                # if the batch has ended
                if totalObservations%batchSize==0:
                    printWrite()
                if totalObservations%statBatchSize==0:
                    WriteStat()
            #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
            printWrite()
            WriteStat()
