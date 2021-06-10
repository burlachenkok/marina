#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys, os, pickle
import psutil 
import time

import_time = time.time()

def serialize(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def deserialize(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def gradientCheck(data, x, f):
    d = max(x.shape)
    g = np.zeros((d,1))
    for i in range(0,d):
        d_step = np.zeros((d,1))
        d_step[i] = 0.001
        g[i] = (f(data, x + d_step) - f(data, x - d_step)) / (2.0 * d_step[i])         # O(eps**2) approximation        
    return g

def printSystemInfo():
    print("")
    print("*********************************************************************************************")
    print("Path to python interpretator:", sys.executable)
    print("Version:", sys.version)
    print("Platform name:", sys.platform)
    print("Physical CPU processors: ", psutil.cpu_count(logical=False))
    print("Logical CPU processors: ", psutil.cpu_count(logical=True))
    print("Current CPU Frequncy: ", psutil.cpu_freq().current, "MHz")
    print("Installed Physical available RAM Memory: %g %s" % (psutil.virtual_memory().total/(1024.0**3), "GBytes"))
    print("Available physical available RAM Memory: %g %s" % (psutil.virtual_memory().available/(1024.0**3), "GBytes"))
    print("")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print("Process Resident Set (Working Set) Size: ", mem_info.rss/(1024.0 * 1024.0), "MBytes")
    print("Virtual Memory used by process: ", mem_info.vms/(1024.0 * 1024.0), "MBytes")
    print("*********************************************************************************************")
    print("Time since program starts: ", str(time.time() - import_time), " seconds")
    print("*********************************************************************************************")
    print("Script name: ", sys.argv[0])
    print("*********************************************************************************************")
