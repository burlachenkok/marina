#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random

class DataSet:
    def trainObservations(self): 
        return self.train_samples.shape[0]
    def trainCases(self):
        return self.trainObservations()
    def trainInstances(self):
        return self.trainObservations()
    def trainRecords(self):
        return self.trainObservations()

    def testObservations(self): 
        return self.test_samples.shape[0]
    def testCases(self):
        return self.testObservations()
    def testInstances(self):
        return self.testObservations()
    def testRecords(self):
        return self.testObservations()
   
    def variables(self): 
        return self.train_samples.shape[1]
    def attributes(self): 
        return variables()
    def fields(self):
        return variables()
    def parametersDimension(self): 
        return variables()

    def __init__(self):
        self.train_samples        = np.zeros( (0, 0), dtype = np.float)
        self.test_samples         = np.zeros( (0, 0), dtype = np.float)
        self.train_true_targets   = np.zeros( (0, 1), dtype = np.float)
        self.test_true_targets    = np.zeros( (0, 1), dtype = np.float)
   
    def analyzeDataset(inputfile):
        f = 0
        s_pos = 0
        s_neg = 0
        i = 0

        with open(inputfile, "r") as f_in:
            for line in f_in:
                line = line.strip("\r\n ").replace('\t', ' ').split(' ')
                line = [item for item in line if len(item)>0]

                if len(line) == 0:
                    continue
            
                if float(line[0]) > 0.0:
                    s_pos = s_pos + 1
                else:
                    s_neg = s_neg + 1

                # With ignoring first letter
                feautes_pos  = [ int( i.split(':')[0] ) for i in line[1:] ]

                # Maximum Feature number
                if len(feautes_pos) > 0:
                    features_max_number = max(feautes_pos)
                    if features_max_number > f:
                        f = features_max_number
                i = i + 1

        return f,s_pos,s_neg

    def readSparseInput(inputfile, features, posEexamples, negExamples, includeBias):
        examples = posEexamples + negExamples

        X = np.zeros((examples,features), float)
        Y = np.zeros((examples,1), float)
        i = 0

        with open(inputfile, "r") as f_in:
            for line in f_in:
                line = line.strip("\r\n ").replace('\t', ' ').split(' ')
                line = [item for item in line if len(item)>0]

                if len(line) == 0:
                    continue
            
                Y[i,0] = float(line[0])
            
                feautes_pos  = [ int( i.split(':')[0] ) for i in line[1:] ]
                feautes_vals = [ float( i.split(':')[1] ) for i in line[1:] ]

                if includeBias:
                    X[i, 0] = 1.0

                for j in range(len(feautes_pos)):
                    # https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q03:_Data_preparation
                    X[i, feautes_pos[j]] = feautes_vals[j]

                i = i + 1

        return X,Y

    def loadDataForClassification(self, test_name, includeBias):
        input_train = "./data/" + test_name
        input_test = input_train + ".t"

        features_train, pos_examples_train, neg_example_train = DataSet.analyzeDataset(input_train)
        features_test, pos_examples_test, neg_example_test = DataSet.analyzeDataset(input_test)

        # Append bias term at first place or zero
        features = max(features_train, features_test) + 1

        Xtrain, Ytrain = DataSet.readSparseInput(input_train,   features, pos_examples_train, neg_example_train, includeBias)
        Xtest,  Ytest  = DataSet.readSparseInput(input_test, features, pos_examples_test, neg_example_test, includeBias)

        self.train_samples        = Xtrain
        self.train_true_targets   = Ytrain       

        self.test_samples       = Xtest
        self.test_true_targets  = Ytest

    def printInfo(self):


        print("Information about dataset")
        print(" number of examples in train set: {0}".format(self.trainObservations()))
        print(" number of examples in test set: {0}".format(self.testObservations()))
        print(" number of features in train/test set: {0}".format(self.variables()))
        print("")
        print("=========================TRAIN SET INFORMATION START====================================")
        print("Postive Targets in Train set: {0}".format(int(np.sum(self.train_true_targets >= 0))))
        print("Negative Targets in Train set: {0}".format(int(np.sum(self.train_true_targets < 0))))
        print("")
        print("Postive Targets in Test set: {0}".format(int(np.sum(self.test_true_targets >= 0))))
        print("Negative Targets in Test set: {0}".format(int(np.sum(self.test_true_targets < 0))))
        print("")
        print("Features characteristics generated (based on train set)")
        for j in range(0, self.variables()):
            print("Feature %02i | minimum %+2.5f | maximum %+2.5f | mean %+2.5f | 1-std.deviation %+2.5f| " % (j, 
                                                                   np.min(self.train_samples[:,j]),
                                                                   np.max(self.train_samples[:,j]),
                                                                   np.mean(self.train_samples[:,j]),
                                                                   (np.mean(self.train_samples[:,j]**2) - np.mean(self.train_samples[:,j])**2)**0.5 
                                                                   ))
        print("")
        print("True Target | minimum %+2.5f | maximum %+2.5f | mean %+2.5f | 1-std.deviation %+2.5f |" % (np.min(self.train_true_targets), np.max(self.train_true_targets),
                                                                                                        np.mean(self.train_true_targets), 
                                                                                                        (np.mean(self.train_true_targets**2) - np.mean(self.train_true_targets)**2)**0.5
                                                                                                        ))
        print("=========================TRAIN SET INFORMATION END====================================")        
        print("")
