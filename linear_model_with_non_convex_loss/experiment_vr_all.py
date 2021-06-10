#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import dataset, compressors, dataset, utils
import time, random, sys, os, math, copy
from mpi4py import MPI

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2

# backup font
plt.rcParams["font.size"] = 27

# For debug purpose
use_grad_check = False                         # For debug purpose (incrase compute time a lot) use numerical gradients
use_first_20_samples = False                   # For debug purpose take only first 10 sample
use_test_set = False                           # Use test set in graphic and calculations

t0 = time.time()

#==================================MPI metainformation===================================================================
comm = MPI.COMM_WORLD
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
mpi_name = MPI.Get_processor_name()
version = MPI.Get_version()
# =================================Configuration of launching============================================================

script_name = os.path.splitext(os.path.basename(__file__))[0]       

#==================================Help functions start==================================================================
def zeroOneLoss(theta, X, Y, margin):
    Z = margin
    Errors = (Z <= 0)
    ErrMean = Errors.mean()
    return ErrMean

def dbgprint(*args):
    printing_dbg = False
    if printing_dbg == True:
        print(f"node {mpi_rank}/{mpi_size}:", *args, flush = True)

def rootprint(*args):
    if mpi_rank == 0:
        print(f"node {mpi_rank}/{mpi_size}:", *args, flush = True)

def get_subset(ctr, indicies):
    return [ctr[ind] for ind in indicies]

#==================================Help functions end====================================================================


rootprint(f"MPI: verions: {version[0]}.{version[1]}")
rootprint("MPI: number of processes in the communicator: ", mpi_size)
rootprint("MPI: rank of this process: ", mpi_rank, ", process name: ", mpi_name)


if mpi_rank == 0:
    utils.printSystemInfo()

#=========================================================================================================================
# GRADIENT AND OBJECTIVE

def lossFunction(theta, X, Y, margin):   
    Z = 1.0 - 1.0/(1.0 + np.exp(-margin))
    ZSquare = Z**2
    res = np.mean(ZSquare)
    return res

def lossFunctionGradient(theta, X, Y, margin):   
    q3 = -2.0 * np.exp(margin - 3*np.log(1+np.exp(margin)))
    #q3 = -2.0 * np.exp(margin)/((np.exp(margin) + 1)**3)
    dbgprint("q3", margin.shape)
   
    n = X.shape[0]
    d = X.shape[1]
    
    G =  np.zeros((n, d))
    for i in range(n):
        gi = Y[i] * q3[i] * X[i,:].T
        G[i, :] = gi

    grad = np.mean(G, axis=0)    
    return grad.reshape(d,1)
    
def regulizer(theta, lamb):
    reg = (np.linalg.norm(theta)**2) * lamb/2.0
    return reg

def regulizerGradient(theta, lamb):
    reg = theta * lamb
    return reg

def computeMargin(X, theta, Y):
    m = np.multiply(X @ theta, Y)
    dbgprint("m", (X @ theta).shape)
    return m
#=========================================================================================================================
# F ON TEST SET
#====================================================================================================================
def ftest(data, lamb, x):
    theta = x
    Xtest = data.test_samples
    Ytest = data.test_true_targets    
    train_margin  = computeMargin(Xtest, theta, Ytest)
    loss = lossFunction(theta, Xtest, Ytest, train_margin)
    reg = 0.0
    if lamb != 0.0:
        reg = regulizer(theta, lamb)
    return loss + reg

#=========================================================================================================================
# F ON TRAIN SET
#====================================================================================================================
def f(data, lamb, x):
    theta = x
    Xtrain = data.train_samples
    Ytrain = data.train_true_targets    
    train_margin  = computeMargin(Xtrain, theta, Ytrain)
    loss = lossFunction(theta, Xtrain, Ytrain, train_margin)
    reg = 0.0
    if lamb != 0.0:
        reg = regulizer(theta, lamb)
    return loss + reg

#=========================================================================================================================
# F FULL GRADIENT
#====================================================================================================================
def fFullGrad(data, lamb, x, train_margin):
    theta = x
    Xtrain = data.train_samples
    Ytrain = data.train_true_targets

    if use_grad_check:
        def testFunction(dataset, x):
            return f(dataset, lamb, x)
        gc = utils.gradientCheck(data, x, testFunction)
        return gc

    if train_margin == None:
        train_margin = computeMargin(Xtrain, theta, Ytrain)
    dbgprint("train_margin", train_margin.shape)
    dbgprint("Xtrain", Xtrain.shape)
    dbgprint("theta", theta.shape)
    dbgprint("Ytrain", Ytrain.shape)

    g = lossFunctionGradient(theta, Xtrain, Ytrain, train_margin)
    if lamb != 0:
        g = g + regulizerGradient(x, lamb)
    dbgprint("g", g.shape)
    return g

#====================================================================================================================
# BATCH GRADIENT OF F IN TRAIN SET
#====================================================================================================================
def fBatchGrad(data, lamb, x, indicies, train_margin):
    theta = x
    Xtrain = data.train_samples[indicies,:]
    Ytrain = data.train_true_targets[indicies,:]

    if use_grad_check:
        def testFunction(dataset, x):
            return f(dataset, lamb, x)
        gc = utils.gradientCheck(data, x, testFunction)
        return gc

    if train_margin == None:
        train_margin = computeMargin(Xtrain, theta, Ytrain)
    dbgprint("train_margin", train_margin.shape)
    dbgprint("Xtrain", Xtrain.shape)
    dbgprint("theta", theta.shape)
    dbgprint("Ytrain", Ytrain.shape)

    g = lossFunctionGradient(theta, Xtrain, Ytrain, train_margin)
    if lamb != 0:
        g = g + regulizerGradient(x, lamb)
    dbgprint("g", g.shape)
    return g

#====================================================================================================================
# BATCH GRADIENT OF F IN TEST SET
#====================================================================================================================
def fFullGradTest(data, lamb, x, test_margin):
    theta = x
    Xtest = data.test_samples
    Ytest = data.test_true_targets

    if test_margin == None:
        test_margin = computeMargin(Xtest, theta, Ytest)

    dbgprint("test_margin", test_margin.shape)
    dbgprint("Xtest", Xtest.shape)
    dbgprint("theta", theta.shape)
    dbgprint("Ytest", Ytest.shape)

    g = lossFunctionGradient(theta, Xtest, Ytest, test_margin)
    if lamb != 0:
        g = g + regulizerGradient(x, lamb)
    dbgprint("g", g.shape)
    return g

#====================================================================================================================
test_name = str(os.getenv("test_name", "w8a"))
KMax        = int(os.getenv("KMax", 40000))                            # maximum number of iterations
KSamplesMax = int(os.getenv("KMax", 40000))                            # samples available in the final statistics
mark_mult = KSamplesMax*1.0/KMax
 
KSamples = [i for i in range(0, KMax, KMax//KSamplesMax)]

fixed_gamma = float(os.getenv("fixed_gamma", 0.5))             # step size for algo
include_bias = bool(os.getenv("include_bias", True))           # include bias into fitting 
fixed_alpha_diana = float(os.getenv("fixed_alpha_diana", 0.0)) # step size for DIANA's for update aggregated shift

use_vr_marina = bool(os.getenv("use_vr_marina", False)) # use VR Marina
use_marina    = bool(os.getenv("use_marina",    False)) # use Marina
use_vr_diana  = bool(os.getenv("use_vr_diana",  False)) # use VR Marina
use_diana     = bool(os.getenv("use_diana",     False)) # use Marina

vr_batch_size_percentage = float(os.getenv("vr_batch_size_percentage", 1.0/100.0)) # VR Marina batch size
p = float(os.getenv("p", 1.0))                        # p metaparameter for algorithm

print("===================================================================")
rootprint("test_name:", test_name)
rootprint("KMax:", KMax)
rootprint("fixed_gamma:", fixed_gamma)
rootprint("include_bias:", include_bias)
rootprint("")
rootprint("use_vr_marina:", use_vr_marina)
rootprint("use_marina:", use_marina)
rootprint("use_vr_diana:", use_vr_diana)
rootprint("use_diana:", use_diana)
rootprint("")
rootprint("Marina specific settings:")
rootprint("vr_batch_size_percentage:", vr_batch_size_percentage)
rootprint("")
rootprint("Diana specific settings:")
rootprint("fixed_alpha_diana:", fixed_alpha_diana)

rootprint("p:", p)
rootprint("===================================================================")

#====================================================================================================================
data = dataset.DataSet()
data.loadDataForClassification(test_name, includeBias = include_bias)

max_target = np.max(data.train_true_targets)
min_target = np.min(data.train_true_targets)


if max_target == min_target:
    print(f"There are only labels of one class in dataset {test_name}. Please fix it.")
    sys.exit(-1)

data.train_true_targets = 2*((data.train_true_targets - min_target) / (max_target - min_target) - 0.5)

max_target = np.max(data.test_true_targets)
min_target = np.min(data.test_true_targets)
data.test_true_targets = 2*((data.test_true_targets - min_target) / (max_target - min_target) - 0.5)

#====================================================================================================================
if use_first_20_samples:
    data.train_samples = data.train_samples[0:20]#000,:]
    data.train_true_targets = data.train_true_targets[0:20]#000,:]
#====================================================================================================================
if mpi_rank == 0:
    data.printInfo()
#====================================================================================================================

D = data.variables()                                           # Dimension of the problem
KNodes = mpi_size                                              # Number of Nodes (each node process it's part of data)
N = data.trainInstances()                                      # Total number of train samples
Workers = KNodes - 1                                           # Worker Nodes
worker_id = mpi_rank - 1                                       # Worker id

if mpi_size == 1:
    #data.train_samples = data.train_samples*0 + 1
    #data.train_true_targets = data.train_true_targets*0 +1
    test_lamba = 0.0

    # Test analytic gradient
    testX = 2*(np.random.uniform(size=(D,1)) - 0.5)
    ga = fFullGrad(data, test_lamba, testX, None)

    def testFunction(dataset, x):
        return f(dataset, test_lamba, x)
    gc = utils.gradientCheck(data, testX, testFunction)

    print(f"relative error for compute gradient in R^{D}: ", np.linalg.norm(gc-ga)/np.linalg.norm(gc))
    print(f"absolute error for compute gradient in R^{D}: ", np.linalg.norm(gc-ga))
    print(f"dot product", np.dot(ga.reshape(1,D), gc.reshape(D,1))/np.linalg.norm(ga)/np.linalg.norm(gc) )   
    sys.exit(0)

SamplePerNodeTrain = math.floor(data.trainObservations() / Workers)   # Samples for each node for train

#=========================================================================================================================
# Obtain need data for node
#=========================================================================================================================
data_for_node = []
Li = []

dbgprint(f"hello {mpi_rank}")
for n in range(Workers):
    data_new = None
    
    # Master Node need all data for final analytic plots
    if n == worker_id or mpi_rank == 0:
        data_new = dataset.DataSet()

        data_new.train_samples       = data.train_samples[n*SamplePerNodeTrain : (n+1)*SamplePerNodeTrain, :]
        data_new.train_true_targets  = data.train_true_targets[n*SamplePerNodeTrain : (n+1)*SamplePerNodeTrain, :]

        data_new.test_samples       = data.test_samples
        data_new.test_true_targets  = data.test_true_targets

        rootprint(f"INFO: Worker {n+1} will process the following number of samples: ", data_new.trainObservations())

        if mpi_rank == 0:
            Li.append(0.15405 * np.max(np.sum((data_new.train_samples)**2,axis=1)))
    data_for_node.append(data_new)

Li = comm.bcast(Li, 0)

Lmax = max(Li)
Lavg = np.mean(Li)
L = Lmax   # Based on paper

rootprint("!Estimation of L smoothnes for f: ", L)
rootprint(f"!Last {(data.trainObservations() - SamplePerNodeTrain * Workers)} records have been ignored")

batch_size_for_worker = math.ceil(vr_batch_size_percentage * SamplePerNodeTrain)
rootprint(f"!INFO: Local dataset for worker is {batch_size_for_worker}")
#======================================== Common params=======================================================
if mpi_rank == 0:
    print("Available workers for 1 master: ", Workers)

t_alg_4_start = time.time()                 # Store start time of algorithm

np.random.seed(35)
x0 = np.random.rand(D,1)*0.0                # start iterate

test_description = []                       # description

KRounds = 1                                 # Number of rounds to average behaviour

ktest_values = [

                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : True, "use_marina": False, "use_vr_diana": False, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(1), D),
                             "component_bits_size": 32
                             }, 

                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : True, "use_marina": False, "use_vr_diana": False, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(5), D),
                             "component_bits_size": 32
                             }, 

                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : True, "use_marina": False, "use_vr_diana": False, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(10), D),
                             "component_bits_size": 32
                             }, 
                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : False, "use_marina": False, "use_vr_diana": True, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(1), D),
                             "component_bits_size": 32
                             }, 

                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : False, "use_marina": False, "use_vr_diana": True, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(5), D),
                             "component_bits_size": 32
                             },
                             {"gamma": fixed_gamma, "p":p, "lamb" : L*(1.0e-6), 
                             "use_vr_marina" : False, "use_marina": False, "use_vr_diana": True, "use_diana": False, "use_gd": False,
                             "init_compressor": lambda cmr: cmr.makeRandKCompressor(int(10), D),
                             "component_bits_size": 32
                             }, 
               ]

one_test = os.getenv("one_test", "").strip()  
rootprint(f"one_test env.variabel: '{one_test}'")  

if len(one_test) > 0 and int(one_test) >= 0:
    ktest_values = [ktest_values[int(one_test)]]

KTests = len(ktest_values)                                   # Total number of tests

#loss_for_train = np.zeros((KTests, KRounds, 1))             # Loss for train

transfered_bits_by_node = np.zeros((KTests, KRounds, KMax)) # Transfered bits
fi_grad_calcs_by_node   = np.zeros((KTests, KRounds, KMax)) # Evaluate number gradients for fi

xk_nodes = np.zeros((KTests, KRounds, KMax, D))
yk_nodes = np.zeros((KTests, KRounds, KMax, D))
xk_nodes_solution = np.zeros((KTests, KRounds, D))

  
# Iterates which describe local models (iterates are stored by rows, each iterate is d-dimensional vector)
for t in range(KTests):
    descr = ktest_values[t]

    #==================================================================================================================================
    # Unpack test description
    #==================================================================================================================================
    i_use_vr_marina = descr["use_vr_marina"]
    i_use_marina    = descr["use_marina"]
    i_use_vr_diana  = descr["use_vr_diana"]
    i_use_diana     = descr["use_diana"]
    i_use_gd        = descr["use_gd"]
    lamb            = descr["lamb"]

    component_bits_size = descr["component_bits_size"]
    compressorForNode = compressors.Compressor()
    compressorForNode.resetStats()
    descr["init_compressor"](compressorForNode)

    #L = 1.0
    w = compressorForNode.getW()
    Ltask = L + lamb

    if i_use_marina:
        # update p for MARINA
        descr["p"] = 1.0/(1+w)#*0.01
        descr["gamma"] = 1.0/(Ltask*(1 + (w*w/Workers)**0.5))
    elif i_use_vr_marina:
        # update p for MARINA
        descr["p"] = min(1.0/(1+w), (batch_size_for_worker)/(batch_size_for_worker + SamplePerNodeTrain))
        descr["gamma"] = 1.0/(Ltask + ((w*Ltask*Ltask + (1+w)*Ltask*Ltask/batch_size_for_worker)**0.5) * ((1 - descr["p"]) / (descr["p"]*Workers))**0.5 ) 
                         #1.0/(Ltask*(1 + (w*w/Workers)**0.5))
    elif i_use_vr_diana or i_use_diana:
        # update alpha for DIANA
        fixed_alpha_diana = 1.0/(1+w)
        m = 1.0
        if i_use_vr_diana:
            m = (SamplePerNodeTrain)/(batch_size_for_worker)

        descr["gamma"] = 1.0/(10*Ltask*(1 + w/Workers)**0.5 * (m**(2.0/3.0) + w + 1))  # Th.4
        descr["p"] = 1/m

        # Convex Settings
        #M = 2*w/(Workers*fixed_alpha_diana)
        #descr["gamma"] = 1.0/( (1 + 2*(w/Workers))*Ltask + M*Ltask*fixed_alpha_diana)  # Covnex case for DIANA

    elif i_use_gd:
        descr["gamma"] = 1.0/Ltask


    gamma = descr["gamma"]          # step size
    p     = descr["p"]              # p for Marina

    rootprint(f"! Compressor for nodes in experiment {t+1}/{KTests} is: ", compressorForNode.name())


    
    specified_algorihms = int(i_use_vr_marina) + int(i_use_marina) + int(i_use_vr_diana) + int(i_use_diana) + int(i_use_gd)

    if specified_algorihms != 1:
        print(f"node {mpi_rank}/{mpi_size}: Before Launch tests please specify only single algorthm! You incorrectly speciy {specified_algorihms}")
        sys.exit(-1)
    #==================================================================================================================================

    rootprint("Evaluation is completed by ", t/KTests * 100.0, "%")
    rootprint(f"  Launch test with {KMax} iterations p={p}, gamma={gamma} across {Workers} workers / test {t+1}:{KTests}")
    
    gprev = None
    gk    = None
    cmd   = None

    #if mpi_rank != 0:
    #    batch_size_for_worker = math.ceil(vr_batch_size_percentage * data_for_node[worker_id].trainObservations())   
    #==================================================================================================================================
    if i_use_diana:
        for r in range(KRounds):
            hk = np.zeros((D,1))          # Shifts for DIANA
            xk_nodes[t, r, 0, :] = x0.T   # Initiate starting point
    
            #if mpi_rank != 0:
            #    batch_size_for_worker = math.ceil(vr_batch_size_percentage * data_for_node[worker_id].trainObservations())
    
            for k in range(0, KMax-1):
                if k % (KMax//10) == 0:
                    rootprint(f"Evaluation for round {r+1}/{KRounds} is completed by ", k/KMax * 100.0, "%")
    
                if mpi_rank == 0:
                    comm.bcast("bcast_xk", root = 0)              # Broadcast xk to all workers
                    xk_nodes[t, r, k, :] = comm.bcast(xk_nodes[t, r, k, :], root = 0)

                    # Aggregate received messages (From paper: (delta^)->gk)
                    gk_avg = np.zeros((D,1))                   
                    for i in range(1, mpi_size):
                       data_mpi = comm.recv(source = i, tag = 0)
                       gk_avg = (gk_avg*(i-1) + data_mpi.reshape(D,1)) / (i)
    
                    # Compute global gradient estimator:
                    #gk = hk + mk_avg
    
                    # Take proximal SGD step
                    xk_nodes[t, r, k + 1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk_avg.reshape(1,D)
    
                    # Update aggregated shift:
                    #hk = hk + fixed_alpha_diana * mk_avg
    
    
                if mpi_rank != 0:
                    cmd = comm.bcast(cmd, root = 0)
                    if cmd == "bcast_xk":
                        xk_nodes[t, r, k, :] = comm.bcast(xk_nodes[t, r, k, :], root = 0)                              # Obtain current iterate from master
                        gk_next = fFullGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k, :].reshape(D,1), None)   # Compute local gradient
                        gk_next = gk_next.reshape(D,1)

                        delta   = gk_next - hk                               # Shifted gradient
                        mk_i    = compressorForNode.compressVector(delta)    # Compress shifted local gradient
                        hk = hk + fixed_alpha_diana * mk_i                  # Update the local shift
                        #comm.send(mk_i.reshape((D,1)), dest = 0, tag = 0)    # Send message to the master (just for numeric purpose we send hk)
                        comm.send(hk.reshape((D,1)), dest = 0, tag = 0)       # Send message to the master (just for numeric purpose we send hk)
                        #===========================================================================
                        fi_grad_calcs_by_node[t,r,k] = 1 * data_for_node[worker_id].trainObservations()
                        transfered_bits_by_node[t,r,k] = compressorForNode.last_need_to_send_advance * component_bits_size
                        #===========================================================================
    
            kSol = np.random.randint(KMax)
            xk_nodes_solution[t, r, :] = xk_nodes[t, r, kSol, :]
    #==================================================================================================================================
    if i_use_gd:
        for r in range(KRounds):
            xk_nodes[t, r, 0, :] = x0.T
            for k in range(0, KMax-1):
                if mpi_rank == 0:
                    gk = fFullGrad(data, lamb, xk_nodes[t, r, k, :].reshape(D,1), None)   
                    xk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk.reshape(1,D) 
                    fi_grad_calcs_by_node[t,r,k] = 1 * data.trainObservations()
                    fi_grad_calcs_by_node[t,r,k] = 1 * data.trainObservations()
    #==================================================================================================================================
    if i_use_vr_diana:
        gprev = None
        gk    = None
        cmd   = None

        for r in range(KRounds):
            hk = np.zeros((D,1))          # Shifts for DIANA
            xk_nodes[t, r, 0, :] = x0.T   # Initiate starting point
    
            #if mpi_rank != 0:
                #batch_size_for_worker = math.ceil(vr_batch_size_percentage * data_for_node[worker_id].trainObservations())
                #print(">>>>>>", batch_size_for_worker)
    
            for k in range(1, KMax):
                if k % (KMax//10) == 0:
                    rootprint(f"Evaluation for round {r+1}/{KRounds} is completed by ", k/KMax * 100.0, "%")
    
                if mpi_rank == 0:
                    # Draw testp Bernoulli random variable (which is equal 1 w.p. p)
                    uk = 0
                    testp = random.random()
    
                    if testp < p:                                          #(1.0/batch_size_for_worker):      # Q: Maybe replace to "P"?
                        uk = 1
                    else:
                        uk = 0
    
                    if uk == 1:
                        comm.bcast("bcast_xk_uk_1", root = 0)              # Broadcast xk to all workers (and uk)
                        comm.bcast(xk_nodes[t, r, k, :], root = 0)
                    if uk == 0:
                        comm.bcast("bcast_xk_uk_0", root = 0)              # Broadcast xk to all workers (and uk)
                        comm.bcast(xk_nodes[t, r, k, :], root = 0)
    
                    # Aggregate received messages (From paper: (delta^)->gk)
                    mk_avg = np.zeros((D,1))                   
                    for i in range(1, mpi_size):
                       data_mpi = comm.recv(source = MPI.ANY_SOURCE, tag = 0)
                       mk_avg = (mk_avg*(i-1) + data_mpi.reshape(D,1)) / (i)
    
                    # Compute global gradient estimator:
                    gk = hk + mk_avg
    
                    # Take proximal SGD step
                    xk_nodes[t, r, k, :] = xk_nodes[t, r, k - 1, :].reshape(1,D) - gamma * gk.reshape(1,D)
    
                    # Update aggregated shift:
                    hk = hk + fixed_alpha_diana * mk_avg
    
    
                if mpi_rank != 0:
                    cmd = comm.bcast(cmd, root = 0)
                    if cmd == "bcast_xk_uk_0" or cmd == "bcast_xk_uk_1":
                        xk_nodes[t, r, k, :] = comm.bcast(xk_nodes[t, r, k, :], root = 0)
    
                        # L-SVRG-US
                        indicies = np.random.permutation(data_for_node[worker_id].trainObservations())[0:batch_size_for_worker]
                        dbgprint(f"sample {len(indicies)} out of {data_for_node[worker_id].trainObservations()}")

                        #jik = np.random.randint(data_for_node[worker_id].trainObservations())
                        #indicies = np.array([jik])
    
                        gk_x = fBatchGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k, :].reshape(D,1), indicies, None) 
                        gk_w = fBatchGrad(data_for_node[worker_id], lamb, yk_nodes[t, r, k, :].reshape(D,1), indicies, None)
                        full_grad_w = fFullGrad(data_for_node[worker_id], lamb, yk_nodes[t, r, k, :].reshape(D,1), None)
    
                        gk_next = gk_x - gk_w + full_grad_w
                        delta   = gk_next - hk
                        mk_i    = compressorForNode.compressVector(delta)    # Compress shifted local gradient
                        comm.send(mk_i.reshape((D,1)), dest = 0, tag = 0)    # Send message to the master
                        hk = hk + fixed_alpha_diana * mk_i                   # Update the local shift
    
                        if k < KMax - 1:
                            if cmd == "bcast_xk_uk_1":  yk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :]
                            if cmd == "bcast_xk_uk_0":  yk_nodes[t, r, k+1, :] = yk_nodes[t, r, k, :]
                        #===========================================================================
                        if k == 0:
                            # need to take full grad at beginning
                            fi_grad_calcs_by_node[t,r,k] = data_for_node[worker_id].trainObservations()    
                        elif cmd == "bcast_xk_uk_1":
                            # update control imply more big gradient at next step
                            fi_grad_calcs_by_node[t,r,k] = data_for_node[worker_id].trainObservations() + 2*indicies.size 
                        else:
                            fi_grad_calcs_by_node[t,r,k] = 2*indicies.size
                        #fi_grad_calcs_by_node[t,r,k] = 0.01
                        transfered_bits_by_node[t,r,k] = compressorForNode.last_need_to_send_advance * component_bits_size
                        #===========================================================================
    
            kSol = np.random.randint(KMax)
            xk_nodes_solution[t, r, :] = xk_nodes[t, r, kSol, :]      
    #==================================================================================================================================
    if i_use_marina or i_use_vr_marina:
        for r in range(KRounds):
            xk_nodes[t, r, 0, :] = x0.T
            gk = fFullGrad(data, lamb, xk_nodes[t, r, 0, :].reshape(D,1), None)
            cmd_prev = ""

            #if mpi_rank != 0:
            #    batch_size_for_worker = math.ceil(vr_batch_size_percentage * data_for_node[worker_id].trainObservations())
    
            for k in range(0, KMax-1):
                #dbgprint("summary--gprev norm", np.linalg.norm(gprev)**2)
    
                if k % (KMax//10) == 0:
                    rootprint(f"Evaluation for round {r+1}/{KRounds} is completed by ", k/KMax * 100.0, "%")
    
                if mpi_rank == 0:
                    # Draw testp Bernoulli random variable (which is equal 1 w.p. p)
                    ck = 0
                    testp = random.random()

                    if testp < p:
#                       rootprint("--!", )
                        ck = 1
                    else:
                        ck = 0
    
                    if ck == 1:
                        dbgprint(">>master node broadcast start")
                        comm.bcast("bcast_g_c1", root = 0)
                        dbgprint(">>master node broadcast end")
                        gk = comm.bcast(gk, root = 0)

                    if ck == 0:
                        dbgprint(">>master node broadcast start")
                        comm.bcast("bcast_g_c0", root = 0)
                        dbgprint(">>master node broadcast end")
                        gk = comm.bcast(gk, root = 0)
                    #===================================================================================================
                    xk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk.reshape(1,D) 
                    dbgprint(f"master.cur iteration {k} gradient: ", np.linalg.norm(gk))
                    dbgprint(f"master--{k}--x--",  xk_nodes[t, r, k, :])
                    dbgprint(f"master--{k}--g--",  gk.reshape(D,1))
                    dbgprint(f"master--{k}--gn--",  np.linalg.norm(gk.reshape(D,1))**2)
    
                    #===================================================================================================
                    g_avg = np.zeros((D,1))
                    for i in range(1, mpi_size):
                       dbgprint(f"--wait {i} process to obtain gradient from it")
                       data_mpi = comm.recv(source = MPI.ANY_SOURCE, tag=0)
                       g_avg = (g_avg*(i-1) + data_mpi.reshape(D,1)) / (i)
                    gk = g_avg
                    #print(np.linalg.norm(fFullGrad(data_for_node[0], lamb, xk_nodes[t, r, k+1, :].reshape(D,1), None) - g_avg))
    
                    #gk = fFullGrad(data_for_node[0], lamb, xk_nodes[t, r, 0, :].reshape(D,1),None)


                if mpi_rank != 0:
                    cmd = comm.bcast(cmd, root = 0)
                    if cmd == "bcast_g_c1":
                        gk = comm.bcast(gk, root = 0)
                        dbgprint(f"woker.cur iteration {k} gradient: ", np.linalg.norm(gk))
    
                        dbgprint(t,r,k)
                        dbgprint(xk_nodes.shape)

#                       xk_nodes[t, r, k, :] = xk_nodes[t, r, k - 1, :].reshape(1,D) - gamma * gprev.reshape(1,D)
#                        gk_next = fFullGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k, :].reshape(D,1), None)   
                        #print(np.linalg.norm(gprev - gk_next))  

               
                        #gprev = gk_next
                        xk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk.reshape(1,D) 
                    
                        #xk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk.reshape(1,D)

                        gk_next_to_send = fFullGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k+1, :].reshape(D,1), None)   

                        dbgprint(f"worker--{k}--x--",  xk_nodes[t, r, k, :])
                        dbgprint(f"worker--{k}--g--",  gk.reshape(D,1))
                        dbgprint(f"worker--{k}--gn--",  np.linalg.norm(gk.reshape(D,1))**2)
    
                        # Case when we send to master really complete gradient
                        transfered_bits_by_node[t,r,k] = D * component_bits_size
                        # On that "c1" mode to evaluate gradient we call oracle number of times how much data is in local "full" batch
                        fi_grad_calcs_by_node[t,r,k] = data_for_node[worker_id].trainObservations()
                        #print(data_for_node[worker_id].trainObservations()) 
                        #print(">>", fi_grad_calcs_by_node[t,r,k])
                        #print(">>>!!!")
                    elif cmd == "bcast_g_c0":
                        gk = comm.bcast(gk, root = 0)
                        xk_nodes[t, r, k+1, :] = xk_nodes[t, r, k, :].reshape(1,D) - gamma * gk.reshape(1,D)
    
                        dbgprint(t,r,k)
                        dbgprint(xk_nodes.shape)
    
                        if i_use_vr_marina:
                            indicies = np.random.permutation(data_for_node[worker_id].trainObservations())[0:batch_size_for_worker]
                            dbgprint(f"sample {len(indicies)} out of {data_for_node[worker_id].trainObservations()}")
                            gk_next = fBatchGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k + 1, :].reshape(D,1), indicies, None)
                            gk_prev = fBatchGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k, :].reshape(D,1), indicies, None)
    
                            # indicies is random subset and it's a very small probabiliy (1.0/C(k,n))^2 that two sample subset will be the same
                            fi_grad_calcs_by_node[t,r,k] = 2 * indicies.size

                        else:
                            gk_next = fFullGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k + 1, :].reshape(D,1), None)
                            gk_prev = fFullGrad(data_for_node[worker_id], lamb, xk_nodes[t, r, k, :].reshape(D,1), None)
                            fi_grad_calcs_by_node[t,r,k] = data_for_node[worker_id].trainObservations()   

                        delta = gk_next - gk_prev
                        dbgprint("detla", delta.shape)
                        gk_next_to_send = gk + compressorForNode.compressVector(delta)
                        #xk_nodes[t, r, k, :] = xk_nodes[t, r, k - 1, :].reshape(1,D) - gamma * gk.reshape(1,D)
    
                        # Case when we send to master really compressed difference
                        transfered_bits_by_node[t,r,k] = compressorForNode.last_need_to_send_advance * component_bits_size
                        #dbgprint("gprev", gprev.shape)
    
                    # Send compress vector to master
                    comm.send(gk_next_to_send.reshape((D,1)), dest = 0, tag = 0)
                    cmd_prev = cmd

            kSol = np.random.randint(KMax)
            xk_nodes_solution[t, r, :] = xk_nodes[t, r, kSol, :]
    
t_alg_4_end = time.time()

rootprint(f"Completed in: ", str(t_alg_4_end - t_alg_4_start), " seconds")

#======================================= Visualize solution==================================================
rootprint(f"INFO: Script '{__file__}' main compute part completed in: ", ((time.time() - t0)/60.0), " minutes")

# Only root node is responsible for plotting
if mpi_rank != 0:
    comm.send(transfered_bits_by_node, dest = 0, tag = 1)
    comm.send(fi_grad_calcs_by_node, dest = 0,   tag = 2)

if mpi_rank == 0:
    transfered_bits_by_nodes    = np.zeros((Workers + 1,KTests,KRounds,KMax))
    fi_grad_calcs_by_nodes      = np.zeros((Workers + 1,KTests,KRounds,KMax))

    for i in range(1,mpi_size):
        status = MPI.Status()      
        data_mpi = comm.recv(source = MPI.ANY_SOURCE, tag = 1, status=status)
        pid = status.Get_source()
        transfered_bits_by_nodes[pid,:,:,:] = data_mpi            

    transfered_bits_by_nodes[0,:,:,:] = transfered_bits_by_node

    for i in range(1,mpi_size):
        status = MPI.Status()      
        data_mpi = comm.recv(source = MPI.ANY_SOURCE, tag = 2, status=status)
        pid = status.Get_source()
        fi_grad_calcs_by_nodes[pid,:,:,:] = data_mpi       #/ (data_for_node[pid-1].trainObservations())           

    # Take batch size from worker-1
    fi_grad_calcs_by_nodes[0,:,:,:] = fi_grad_calcs_by_node # /data_for_node[0].trainObservations()

    main_fig_p1      = plt.figure(figsize=(12, 8))
    main_fig_p2      = plt.figure(figsize=(12, 8))

    oracles_fig_p1   = plt.figure(figsize=(12, 8))
    oracles_fig_p2   = plt.figure(figsize=(12, 8))

    transport_fig_p1 = plt.figure(figsize=(12, 8))
    transport_fig_p2 = plt.figure(figsize=(12, 8))

    aux_fig_p1       = plt.figure(figsize=(12, 8))
    aux_fig_p2       = plt.figure(figsize=(12, 8))

    main_description = ""
    aux_description = ""

    for t in range(KTests):
        descr = ktest_values[t]

        # Unpack configuration used to launch specific test
        gamma = descr["gamma"]
        p     = descr["p"]
        lamb  = descr["lamb"]
        component_bits_size = descr["component_bits_size"]   
        i_use_vr_marina = descr["use_vr_marina"]
        i_use_marina    = descr["use_marina"]
        i_use_vr_diana  = descr["use_vr_diana"]
        i_use_diana     = descr["use_diana"]
        i_use_gd       = descr["use_gd"]

        specified_algorihms = int(i_use_vr_marina) + int(i_use_marina) + int(i_use_vr_diana) + int(i_use_diana) + int(i_use_gd)
    
        if specified_algorihms != 1:
            print(f"node {mpi_rank}/{mpi_size}: Please specify only single algorthm! You incorrectly speciy {specified_algorihms}")
            sys.exit(-1)

        compr = compressors.Compressor()
        descr["init_compressor"](compr)
        compressor_name = compr.name()

        transfered_bits_dx = np.zeros(KMax)
        transfered_bits_mx = np.zeros(KMax)

        fi_grad_calcs_mx = np.zeros(KMax)
        fi_grad_calcs_dx = np.zeros(KMax)

        # transfered_bits_by_nodes and fi_grad_calcs_by_nodes has shape: ((Workers,KTests,KRounds,KMax))
        for z in range(KMax):
            transfered_bits_mx[z] = np.mean(transfered_bits_by_nodes[:,t,0,z])
            transfered_bits_dx[z] = np.mean( (transfered_bits_by_nodes[:,t,0,z] - transfered_bits_mx[z])**2)

        for z in range(KMax):
            fi_grad_calcs_mx[z] = np.mean(fi_grad_calcs_by_nodes[:,t,0,z])
            fi_grad_calcs_dx[z] = np.mean( (fi_grad_calcs_by_nodes[:,t,0,z] - fi_grad_calcs_mx[z])**2)

        transfered_bits_mean = np.sum(transfered_bits_by_nodes[:,t, 0, :], axis = 0) / Workers
        fi_grad_calcs_sum   = np.sum(fi_grad_calcs_by_nodes[:,t, 0, :], axis = 0)        

        for i in range(1, KMax):
            transfered_bits_mean[i] = transfered_bits_mean[i] + transfered_bits_mean[i-1]

        for i in range(1, KMax):
            fi_grad_calcs_sum[i] = fi_grad_calcs_sum[i] + fi_grad_calcs_sum[i-1]

        fn_train_with_regul_loss = []
        fn_train_loss = []
        fn_test_loss = []

        fn_train_with_regul_loss_grad_norm = []
        fn_train_loss_grad_norm = []
        fn_test_loss_grad_norm = []

        rootprint(f"Generate summary for test {t+1}/{KTests}")
        for k in KSamples:
            if k % (KMax//10) == 0:
                rootprint(f" summary is completed by ", k/KMax * 100.0, "%")

            fn_train_with_regulizer = []
            #fn_train = []
            fn_test = []

            fn_train_with_regulizer_grad = []
            #fn_train_grad = []
            fn_test_grad = []

            # Calculate across workers
            for n in range(Workers):
                xcur = xk_nodes[t, 0, k, :].reshape(D,1)

                fn_train_with_regulizer.append( f(data_for_node[n], lamb, xcur) )
                #fn_train.append( f(data_for_node[n], 0.0, xcur) )

                #if use_test_set:
                #    fn_test.append( ftest(data_for_node[n], 0.0, xcur) )

                fn_train_with_regulizer_grad.append( np.linalg.norm(fFullGrad(data_for_node[n], lamb, xcur, None))**2 )
                #fn_train_grad.append( np.linalg.norm(fFullGrad(data_for_node[n], 0.0, xcur, None))**2 )

                #if use_test_set:   
                #    fn_test_grad.append( np.linalg.norm(fFullGradTest(data_for_node[n], 0.0, xcur, None))**2 )

                dbgprint(f"summary-{k}--xcur", xcur)
                #dbgprint(f"summary-{k}--gn", fn_train_grad[-1])

            # Mean across shard            
            fn_train_with_regulizer_mean = np.mean(fn_train_with_regulizer)
            #fn_train_mean = np.mean(fn_train)
            #if use_test_set:   
            #    fn_test_mean = np.mean(fn_test)
            fn_train_with_regulizer_grad_mean = np.mean(fn_train_with_regulizer_grad)
            #fn_train_grad_mean = np.mean(fn_train_grad)
            #if use_test_set:   
            #    fn_test_grad_mean = np.mean(fn_test_grad)

            # Final arrays 
            fn_train_with_regul_loss.append(fn_train_with_regulizer_mean)
            #fn_train_loss.append(fn_train_mean)
            #if use_test_set:   
            #    fn_test_loss.append(fn_test_mean)
            fn_train_with_regul_loss_grad_norm.append(fn_train_with_regulizer_grad_mean)
            #fn_train_loss_grad_norm.append(fn_train_grad_mean)
            #if use_test_set:   
            #    fn_test_loss_grad_norm.append(fn_test_grad_mean)

        dbgprint("fn_train_with_regul_loss", len(fn_train_with_regul_loss))
        #dbgprint("fn_train_loss", len(fn_train_loss))
        #if use_test_set:   
        #    dbgprint("fn_test", len(fn_test))

        # Serialize experiment info
        my = {}
        my["fn_train_with_regul_loss"] = fn_train_with_regul_loss
        #my["fn_train_loss"] = fn_train_loss
        my["fn_test_loss"] = fn_test_loss
        my["fn_train_with_regul_loss_grad_norm"] = fn_train_with_regul_loss_grad_norm
        #my["fn_train_loss_grad_norm"] = fn_train_loss_grad_norm
        my["fn_test_loss_grad_norm"] = fn_test_loss_grad_norm
        #my["xk_nodes"] = xk_nodes[t:t+1,...]
        #my["xk_nodes_sampled"] = xk_nodes[t:t+1, :, KSamples, :]

        my["gamma"] = gamma
        my["p"] = p
        my["KMax"] = KMax
        my["KSamplesMax"] = KSamplesMax
        my["KSamples"] = KSamples
        my["lamb"] = lamb
        my["include_bias"] = include_bias

        my["vr_batch_size_percentage"] = vr_batch_size_percentage
        my["component_bits_size"] = component_bits_size
        my["i_use_vr_marina"] = i_use_vr_marina
        my["i_use_marina"]    = i_use_marina
        my["i_use_vr_diana"]  = i_use_vr_diana
        my["i_use_diana"]     = i_use_diana
        my["i_use_gd"]        = i_use_gd

        my["transfered_bits_by_nodes"] = transfered_bits_by_nodes
        my["fi_grad_calcs_by_nodes"]   = fi_grad_calcs_by_nodes
        my["fixed_alpha_diana"]        = fixed_alpha_diana
        prefix4algo = ""

        my["descr"] = descr

        ctest = compressors.Compressor()
        descr["init_compressor"](ctest)
        my["descr"]["init_compressor"] = ctest.fullName()

        if hasattr(ctest, "K"):
            my["descr"]["K"] = ctest.K

        my["KMax"] = KMax
        my["workers"] = Workers
        my["sample_per_node_train"] = SamplePerNodeTrain
        my["train_set_size"]  = data.trainObservations()
        my["train_variables"] = data.variables()

        if i_use_vr_diana or i_use_vr_marina:
            my["brpime_batch_size_for_worker_in_local_samples"] = batch_size_for_worker
            my["m_batch_size_for_worker_in_percentage_to_local_data"] = vr_batch_size_percentage * 100.0
        else:
            my["bprime_batch_size_for_worker_in_local_samples"] = SamplePerNodeTrain
            my["m_batch_size_for_worker_in_percentage_to_local_data"] = 1.0 * 100.0

        my["fixed_alpha_diana"] = fixed_alpha_diana
        my["dataset"] = test_name

        if i_use_vr_marina: prefix4algo = "vr_marina"
        if i_use_marina:    prefix4algo = "marina"
        if i_use_vr_diana:  prefix4algo = "vr_diana"
        if i_use_diana:     prefix4algo = "diana"
        if i_use_gd:        prefix4algo = "gd"

        if len(one_test) == 0:
            utils.serialize(my, f"experiment_{t}_{prefix4algo}_{test_name}.bin")
        else:
            utils.serialize(my, f"experiment_{one_test}_{prefix4algo}_{test_name}.bin")
        #===============================================================================
        rootprint(f"final iterate for {t+1}/{KTests} test. it's l2 norm square divided by two: ", (np.linalg.norm(xk_nodes[t,0,KMax-1,:])**2)/2.0)
        rootprint(f"final iterate for {t+1}/{KTests} test. it's l2 norm: ", np.linalg.norm(xk_nodes[t,0,KMax-1,:]))
        #===============================================================================
        # Figures
        #===============================================================================
        algo_name = ""
        short_algo_name =  ""

        if i_use_vr_marina: algo_name=f'VR-MARINA [$\\gamma$={gamma:g}, p={p:g}, batch={vr_batch_size_percentage*100:g}% of local data]. Compress: {compressor_name}'
        if i_use_marina:    algo_name=f'MARINA [$\\gamma$={gamma:g}, p={p:g}]. Compress: {compressor_name}'
        if i_use_diana:     algo_name=f'DIANA  [$\\gamma$={gamma:g}, $\\alpha$={fixed_alpha_diana}]. Compress: {compressor_name}'
        if i_use_vr_diana:  algo_name=f'VR-DIANA [$\\gamma$={gamma:g},$\\alpha$={fixed_alpha_diana}]. Compressor: {compressor_name}'
        if i_use_gd:        algo_name=f'GD [$\\gamma$={gamma:g}]'


        if i_use_vr_marina: short_algo_name=f'VR-MARINA {compressor_name}'
        if i_use_marina:    short_algo_name=f'MARINA {compressor_name}'
        if i_use_diana:     short_algo_name=f'DIANA {compressor_name}'
        if i_use_vr_diana:  short_algo_name=f'VR-DIANA {compressor_name}'
        if i_use_gd:        short_algo_name=f'GD'

        markevery = [ int(mark_mult*KMax/10.0), int(mark_mult*KMax/13.0), int(mark_mult*KMax/7.0), int(mark_mult*KMax/9.0), int(mark_mult*KMax/11.0), 
                    int(mark_mult*KMax/13.0) ][t%6]
        marker = ["x","^","*","x","^","*",][t%6]
        color = ["#e41a1c", "#377eb8", "#4daf4a", "#e41a1c", "#377eb8", "#4daf4a"][t%6]
        linestyle = ["solid", "solid", "solid", "dashed","dashed","dashed"][t%6]

        #===================================================================================================================================
        fig = plt.figure(main_fig_p1.number)
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogy(KSamples, fn_train_with_regul_loss, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            ax.set_xlabel('Iteration', fontdict = {'fontsize':35})
            ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


        fig = plt.figure(main_fig_p2.number)
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogy(KSamples, fn_train_with_regul_loss_grad_norm, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            ax.set_xlabel('Iteration', fontdict = {'fontsize':35})
            ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)            
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)

        #===================================================================================================================================
        fig = plt.figure(transport_fig_p1.number)
        ax = fig.add_subplot(1, 1, 1)
        if not i_use_gd:
            ax.semilogy(get_subset(transfered_bits_mean,KSamples), fn_train_with_regul_loss, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            ax.set_xlabel(f'#bits/n', fontdict = {'fontsize':35})
            ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


        fig = plt.figure(transport_fig_p2.number)
        ax = fig.add_subplot(1, 1, 1)
        if not i_use_gd:
            ax.semilogy(get_subset(transfered_bits_mean,KSamples), fn_train_with_regul_loss_grad_norm, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            ax.set_xlabel(f'#bits/n', fontdict = {'fontsize':35})
            ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


        #=====================================================================================================================================
        fig = plt.figure(oracles_fig_p1.number)
        ax = fig.add_subplot(1, 1, 1)
        epochs = (fi_grad_calcs_sum * 1.0) / (Workers * SamplePerNodeTrain)

        ax.semilogy(get_subset(epochs,KSamples), fn_train_with_regul_loss, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            ax.set_xlabel(f'# epochs',fontdict = {'fontsize':35})
            ax.set_ylabel('$f(x)$',fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


        fig = plt.figure(oracles_fig_p2.number)
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogy(get_subset(epochs,KSamples), fn_train_with_regul_loss_grad_norm, color=color, marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
        if t == KTests - 1:
            main_description = main_description + algo_name
            ax.set_xlabel(f'# epochs', fontdict = {'fontsize':35})
            ax.set_ylabel('$||\\nabla f(x)||^2$',fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
            plt.title(f'{test_name}', fontdict = {'fontsize':35})
            rootprint("INFO: PLOT NAME:", f'"{test_name}", n={Workers}, d={D}; ' + main_description)
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)

        else:
            main_description = main_description + algo_name + ", "
        #ax.semilogy([0, 1], [0.25, 0.25], color='black', lw=2, transform = plt.gcf().transFigure, clip_on = False)
        #=====================================================================================================================================
        fig = plt.figure(aux_fig_p1.number)
        ax = fig.add_subplot(1, 1, 1)
        #ax.fill_between(range(1,KMax), fi_grad_calcs_mx[1:] - 3*(fi_grad_calcs_dx[1:]**0.5), fi_grad_calcs_mx[1:] + 3*(fi_grad_calcs_dx[1:]**0.5), color='#539ecd')
        ax.plot(range(0,KMax-1), fi_grad_calcs_mx[:-1], color=color, marker=marker,  markevery=markevery, linestyle=linestyle, label=short_algo_name)

        #'Oracle gradient calculation per iteration
        if t == KTests - 1:
            ax.set_xlabel('Iteration', fontdict = {'fontsize':35})
            ax.set_ylabel('Oracle request for evaluate $\\nabla f_i(x)$ at iteration',fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)
#            fig.suptitle(f'{test_name}')
            plt.title(f'{test_name}', fontdict = {'fontsize' : 35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


        fig = plt.figure(aux_fig_p2.number)
        ax = fig.add_subplot(1, 1, 1)

        #ax.fill_between(range(1,KMax), transfered_bits_mx[1:] - 3*(transfered_bits_dx[1:]**0.5), transfered_bits_mx[1:] + 3*(transfered_bits_dx[1:]**0.5), color='#539ecd')
        ax.plot(range(0,KMax-1), transfered_bits_mx[:-1], color=color, marker=marker,  markevery=markevery, linestyle=linestyle, label=short_algo_name)

        # Send bits per iteration
        if t == KTests - 1:
            ax.set_xlabel('Iteration', fontdict = {'fontsize':35})
            ax.set_ylabel('Sent bits from node to master at iteration', fontdict = {'fontsize':35})
            ax.grid(True)
            ax.legend(loc='best', fontsize = 25)            
#           fig.suptitle(f'{test_name}')
            plt.title(f'{test_name}', fontdict = {'fontsize' : 35})
            plt.xticks(fontsize=27)
            plt.yticks(fontsize=30)


    main_fig_p1.tight_layout()
    main_fig_p2.tight_layout()
    oracles_fig_p1.tight_layout()
    oracles_fig_p1.tight_layout()

    rootprint(f"INFO: Script '{__file__}' totally completed in: ", ((time.time() - t0)/60.0), " minutes")
    #plt.show()

    save_to = script_name + "_" + os.path.basename(test_name) + "_main"+ "_p1" + "_" + one_test + ".pdf"
    main_fig_p1.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_main"+ "_p2" + "_" + one_test + ".pdf"
    main_fig_p2.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_oracles" "_p1" + "_" + one_test + ".pdf"
    oracles_fig_p1.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_oracles" "_p2" + "_" + one_test + ".pdf"
    oracles_fig_p2.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_transport" + "_p1" + "_" + one_test + ".pdf"
    transport_fig_p1.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_transport" + "_p2" + "_" + one_test + ".pdf"
    transport_fig_p2.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 
 
    save_to = script_name + "_" + os.path.basename(test_name) + "_aux" + "_p1" + "_" + one_test + ".pdf"
    aux_fig_p1.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

    save_to = script_name + "_" + os.path.basename(test_name) + "_aux" + "_p2" + "_" + one_test + ".pdf"
    aux_fig_p2.savefig(save_to, bbox_inches='tight')
    rootprint("Image is saved into: ", save_to) 

