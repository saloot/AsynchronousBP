#=======================DEFAULT VALUES FOR THE VARIABLES=======================

#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import os
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
import random
import copy
#==============================================================================



#================================INITIALIZATIONS===============================
N = 128                                             # Codeword length
K = 64                                              # Messageword length
d_v = 4                                             # Degree of variable nodes
d_c = 8                                             # Degree of check nodes
d_max = 1                                           # Maximum delay of the edges in the parity check matrix. A value of 0 represents standard belief propagation
no_avg_itrs = 500                                   # Number of times a random noisy vector is generated for decoding
E = N * d_v                                         # Number of edges in the graph
if not os.path.isdir('./Plots'):                    # Create a folder if not already exists for plotting files
    os.makedirs('./Plots')

e0_considered_time = np.array([1,3,5,7,9,13,17,21,25,42,46,50,54,58,62])     # The set of erasures that will be used to plot 'Time vs. No. Corrected Bits' plot
e0_for_deg_range = np.array([16,32,64])             # The number of erasures that will used to plot the evolution of degree one nodes
max_decoding_itr = 100000                            # Maximum number of iterations spent by different decoders
track_deg_one_flag = 0
#===============================================================================



#========================READ AND COMBINE THE RESULTS===========================


#-------------------ITR for Asynchronous Non-Uniform Decoding------------------
file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_ITR_asynch_non_uniform = read_vals[:,1]

e0_uniq = np.unique(e0)
ITR_asynch_non_uniform = np.zeros([len(e0_uniq)])
ITR_asynch_non_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_ITR_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_asynch_non_uniform[ind] = ITR_asynch_non_uniform[ind] + raw_ITR_asynch_non_uniform[i]#/float(e*no_avg_itrs-raw_BER_asynch_non_uniform[i]*N*no_avg_itrs)
    no_instances[ind] = no_instances[ind] + 1

ITR_asynch_non_uniform = np.divide(ITR_asynch_non_uniform,no_instances)
ITR_asynch_non_uniform = ITR_asynch_non_uniform/float(E)

for i in range(0,len(raw_ITR_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_asynch_non_uniform_std[ind] = ITR_asynch_non_uniform_std[ind] + pow((raw_ITR_asynch_non_uniform[i]/float(E))-ITR_asynch_non_uniform[ind],2)
    
ITR_asynch_non_uniform_std = pow(np.divide(ITR_asynch_non_uniform_std,no_instances),0.5)

file_name = "./Plots/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),ITR_asynch_non_uniform.T,ITR_asynch_non_uniform_std.T]).T,'%f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

#---------------------ITR for Asynchronous Uniform Decoding--------------------
file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_ITR_asynch_uniform = read_vals[:,1]

e0_uniq = np.unique(e0)
ITR_asynch_uniform = np.zeros([len(e0_uniq)])
ITR_asynch_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_ITR_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_asynch_uniform[ind] = ITR_asynch_uniform[ind] + raw_ITR_asynch_uniform[i]#/float(e*no_avg_itrs-raw_BER_asynch_uniform[i]*N*no_avg_itrs
    no_instances[ind] = no_instances[ind] + 1

ITR_asynch_uniform = np.divide(ITR_asynch_uniform,no_instances)
ITR_asynch_uniform = ITR_asynch_uniform/float(E)

for i in range(0,len(raw_ITR_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_asynch_uniform_std[ind] = ITR_asynch_uniform_std[ind] + pow((raw_ITR_asynch_uniform[i]/float(E))-ITR_asynch_uniform[ind],2)
    
ITR_asynch_uniform_std = pow(np.divide(ITR_asynch_uniform_std,no_instances),0.5)
file_name = "./Plots/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),ITR_asynch_uniform.T,ITR_asynch_uniform_std.T]).T,'%f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------
    
#---------------------ITR for Synchronous Uniform Decoding----------------------
file_name = "./Results/ITR_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_ITR_synch = read_vals[:,1]

e0_uniq = np.unique(e0)
ITR_synch = np.zeros([len(e0_uniq)])
ITR_synch_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_ITR_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_synch[ind] = ITR_synch[ind] + raw_ITR_synch[i]#/float(e*no_avg_itrs-raw_BER_synch[i]*N*no_avg_itrs)
    no_instances[ind] = no_instances[ind] + 1

ITR_synch = np.divide(ITR_synch,no_instances)
ITR_synch = ITR_synch/float(E)

for i in range(0,len(raw_ITR_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    ITR_synch_std[ind] = ITR_synch_std[ind] + pow((raw_ITR_synch[i]/float(E))-ITR_synch[ind],2)
    
ITR_synch_std = pow(np.divide(ITR_synch_std,no_instances),0.5)

file_name = "./Plots/ITR_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),ITR_synch.T,ITR_synch_std.T]).T,'%f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------

#-------------------BER for Asynchronous Non-Uniform Decoding-------------------
file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_BER_asynch_non_uniform = read_vals[:,1]

e0_uniq = np.unique(e0)
BER_asynch_non_uniform = np.zeros([len(e0_uniq)])
BER_asynch_non_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_BER_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_asynch_non_uniform[ind] = BER_asynch_non_uniform[ind] + raw_BER_asynch_non_uniform[i]
    no_instances[ind] = no_instances[ind] + 1

BER_asynch_non_uniform = np.divide(BER_asynch_non_uniform,no_instances)
BER_asynch_non_uniform = BER_asynch_non_uniform + 1e-8

for i in range(0,len(raw_BER_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_asynch_non_uniform_std[ind] = BER_asynch_non_uniform_std[ind] + pow(raw_BER_asynch_non_uniform[i]-BER_asynch_non_uniform[ind],2)


BER_asynch_non_uniform_std = pow(np.divide(BER_asynch_non_uniform_std,no_instances),0.5)
file_name = "./Plots/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),BER_asynch_non_uniform.T,BER_asynch_non_uniform_std.T]).T,'%3.9f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

#---------------------BER for Asynchronous Uniform Decoding--------------------
file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_BER_asynch_uniform = read_vals[:,1]

e0_uniq = np.unique(e0)
BER_asynch_uniform = np.zeros([len(e0_uniq)])
BER_asynch_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_BER_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_asynch_uniform[ind] = BER_asynch_uniform[ind] + raw_BER_asynch_uniform[i]
    no_instances[ind] = no_instances[ind] + 1

BER_asynch_uniform = np.divide(BER_asynch_uniform,no_instances)
BER_asynch_uniform = BER_asynch_uniform+ 1e-8

for i in range(0,len(raw_BER_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_asynch_uniform_std[ind] = BER_asynch_uniform_std[ind] + pow(raw_BER_asynch_uniform[i]-BER_asynch_uniform[ind],2)


BER_asynch_uniform_std = pow(np.divide(BER_asynch_uniform_std,no_instances),0.5)
file_name = "./Plots/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),BER_asynch_uniform.T,BER_asynch_uniform_std.T]).T,'%3.9f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

#-------------------------BER for Synchronous Decoding-------------------------
file_name = "./Results/BER_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_BER_synch = read_vals[:,1]

e0_uniq = np.unique(e0)
BER_synch = np.zeros([len(e0_uniq)])
BER_synch_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_BER_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_synch[ind] = BER_synch[ind] + raw_BER_synch[i]
    no_instances[ind] = no_instances[ind] + 1

BER_synch = np.divide(BER_synch,no_instances)
BER_synch = BER_synch + 1e-8

for i in range(0,len(raw_BER_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    BER_synch_std[ind] = BER_synch_std[ind] + pow(raw_BER_synch[i]-BER_synch[ind],2)


BER_synch_std = pow(np.divide(BER_synch_std,no_instances),0.5)

file_name = "./Plots/BER_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),BER_synch.T,BER_synch_std.T]).T,'%3.9f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------


#-------------------TIME for Asynchronous Non-Uniform Decoding------------------
file_name = "./Results/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_TIME_asynch_non_uniform = read_vals[:,1]
raw_TIME_asynch_non_uniform = np.divide(raw_TIME_asynch_non_uniform,2)

e0_uniq = np.unique(e0)
TIME_asynch_non_uniform = np.zeros([len(e0_uniq)])
TIME_asynch_non_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_TIME_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_asynch_non_uniform[ind] = TIME_asynch_non_uniform[ind] + raw_TIME_asynch_non_uniform[i]
    no_instances[ind] = no_instances[ind] + 1

TIME_asynch_non_uniform = np.divide(TIME_asynch_non_uniform,no_instances)

for i in range(0,len(raw_TIME_asynch_non_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_asynch_non_uniform_std[ind] = TIME_asynch_non_uniform_std[ind] + pow(raw_TIME_asynch_non_uniform[i]-TIME_asynch_non_uniform[ind],2)
    
TIME_asynch_non_uniform_std = pow(np.divide(TIME_asynch_non_uniform_std,no_instances),0.5)

file_name = "./Plots/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),TIME_asynch_non_uniform.T,TIME_asynch_non_uniform_std.T]).T,'%f',delimiter='\t',newline='\n')


file_name = "./Plots/TIME_vs_Bits_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_X_" + str(max_decoding_itr)+".txt"
vals = (e0_uniq - N*BER_asynch_non_uniform).astype(int)
np.savetxt(file_name,np.vstack([vals.T,TIME_asynch_non_uniform.T]).T,'%f',delimiter='\t',newline='\n')

for ee in e0_considered_time:
    file_name = "./Plots/TIME_vs_Bits_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_e_" + str(ee) + "_X_" + str(max_decoding_itr)+".txt"
    ind = list(e0_uniq).index(ee)
    vals = (ee - N*BER_asynch_non_uniform[ind]).astype(int)
    np.savetxt(file_name,np.reshape(np.array([vals,TIME_asynch_non_uniform[ind]]),[1,2]),'%f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

#--------------------TIME for Asynchronous Uniform Decoding--------------------
file_name = "./Results/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_TIME_asynch_uniform = read_vals[:,1]
raw_TIME_asynch_uniform = np.divide(raw_TIME_asynch_uniform,2)

e0_uniq = np.unique(e0)
TIME_asynch_uniform = np.zeros([len(e0_uniq)])
TIME_asynch_uniform_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_TIME_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_asynch_uniform[ind] = TIME_asynch_uniform[ind] + raw_TIME_asynch_uniform[i]
    no_instances[ind] = no_instances[ind] + 1

TIME_asynch_uniform = np.divide(TIME_asynch_uniform,no_instances)

for i in range(0,len(raw_TIME_asynch_uniform)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_asynch_uniform_std[ind] = TIME_asynch_uniform_std[ind] + pow(raw_TIME_asynch_uniform[i]-TIME_asynch_uniform[ind],2)
    
TIME_asynch_uniform_std = pow(np.divide(TIME_asynch_uniform_std,no_instances),0.5)

file_name = "./Plots/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),TIME_asynch_uniform.T,TIME_asynch_uniform_std.T]).T,'%f',delimiter='\t',newline='\n')

file_name = "./Plots/TIME_vs_Bits_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_X_" + str(max_decoding_itr)+".txt"
vals = (e0_uniq - N*BER_asynch_uniform).astype(int)
np.savetxt(file_name,np.vstack([vals.T,TIME_asynch_uniform.T]).T,'%f',delimiter='\t',newline='\n')

for ee in e0_considered_time:
    file_name = "./Plots/TIME_vs_Bits_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + "_e_" + str(ee) + "_X_" + str(max_decoding_itr)+".txt"
    ind = list(e0_uniq).index(ee)
    vals = (ee - N*BER_asynch_uniform[ind]).astype(int)
    np.savetxt(file_name,np.reshape(np.array([vals,TIME_asynch_uniform[ind]]),[1,2]),'%f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------
    
#---------------------TIME for Asynchronous Uniform Decoding--------------------
file_name = "./Results/TIME_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
e0 = read_vals[:,0]
raw_TIME_synch = read_vals[:,1]
raw_TIME_synch = np.divide(raw_TIME_synch,2)

e0_uniq = np.unique(e0)
TIME_synch = np.zeros([len(e0_uniq)])
TIME_synch_std = np.zeros([len(e0_uniq)])
no_instances = np.zeros([len(e0_uniq)])
for i in range(0,len(raw_TIME_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_synch[ind] = TIME_synch[ind] + raw_TIME_synch[i]#/float(e*no_avg_itrs-raw_BER_synch[i]*N*no_avg_itrs)
    no_instances[ind] = no_instances[ind] + 1

TIME_synch = np.divide(TIME_synch,no_instances)


for i in range(0,len(raw_TIME_synch)):
    e = e0[i]
    ind = list(e0_uniq).index(e)
    TIME_synch_std[ind] = TIME_synch_std[ind] + pow(raw_TIME_synch[i]-TIME_synch[ind],2)
    
TIME_synch_std = pow(np.divide(TIME_synch_std,no_instances),0.5)

file_name = "./Plots/TIME_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
np.savetxt(file_name,np.vstack([e0_uniq/float(N),TIME_synch.T,TIME_synch_std.T]).T,'%f',delimiter='\t',newline='\n')

file_name = "./Plots/TIME_vs_Bits_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_X_" + str(max_decoding_itr)+".txt"
vals = (e0_uniq - N*BER_synch).astype(int)
np.savetxt(file_name,np.vstack([vals.T,TIME_synch.T]).T,'%f',delimiter='\t',newline='\n')


for ee in e0_considered_time:
    file_name = "./Plots/TIME_vs_Bits_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_e_" + str(ee) + "_X_" + str(max_decoding_itr)+".txt"
    ind = list(e0_uniq).index(ee)
    vals = (ee - N*BER_synch[ind]).astype(int)
    np.savetxt(file_name,np.reshape(np.array([vals,TIME_synch[ind]]),[1,2]),'%f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------



if track_deg_one_flag: 
    
    for e0_for_deg in e0_for_deg_range:
    
        #-------------------TIME for Asynchronous Non-Uniform Decoding------------------
        file_name = "./Results/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"
        read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
        e0 = read_vals[:,0]
        raw_DEG_asynch_non_uniform = read_vals[:,1]

        itr_uniq = np.unique(e0)
        DEG_asynch_non_uniform = np.zeros([len(itr_uniq)])
        no_instances = np.zeros([len(itr_uniq)])
        for i in range(0,len(raw_DEG_asynch_non_uniform)):
            e = e0[i]
            ind = list(itr_uniq).index(e)
            DEG_asynch_non_uniform[ind] = DEG_asynch_non_uniform[ind] + raw_DEG_asynch_non_uniform[i]
            no_instances[ind] = no_instances[ind] + 1

        DEG_asynch_non_uniform = np.divide(DEG_asynch_non_uniform,no_instances)
        file_name = "./Plots/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"
        np.savetxt(file_name,np.vstack([itr_uniq,DEG_asynch_non_uniform.T]).T,'%f',delimiter='\t',newline='\n')
        #------------------------------------------------------------------------------

        #--------------------TIME for Asynchronous Uniform Decoding--------------------
        file_name = "./Results/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"
        read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
        e0 = read_vals[:,0]
        raw_DEG_asynch_uniform = read_vals[:,1]
    
        itr_uniq = np.unique(e0)
        DEG_asynch_uniform = np.zeros([len(itr_uniq)])
        no_instances = np.zeros([len(itr_uniq)])
        for i in range(0,len(raw_DEG_asynch_uniform)):
            e = e0[i]
            ind = list(itr_uniq).index(e)
            DEG_asynch_uniform[ind] = DEG_asynch_uniform[ind] + raw_DEG_asynch_uniform[i]
            no_instances[ind] = no_instances[ind] + 1

        DEG_asynch_uniform = np.divide(DEG_asynch_uniform,no_instances)
        file_name = "./Plots/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_0_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"
        np.savetxt(file_name,np.vstack([itr_uniq,DEG_asynch_uniform.T]).T,'%f',delimiter='\t',newline='\n')
        #-------------------------------------------------------------------------------
    
        #---------------------TIME for Asynchronous Uniform Decoding--------------------
        file_name = "./Results/Deg_One_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"    
        read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
        e0 = read_vals[:,0]
        raw_DEG_synch = read_vals[:,1]

        itr_uniq = np.unique(e0)
        DEG_synch = np.zeros([len(itr_uniq)])
        no_instances = np.zeros([len(itr_uniq)])
        for i in range(0,len(raw_DEG_synch)):
            e = e0[i]
            ind = list(itr_uniq).index(e)
            DEG_synch[ind] = DEG_synch[ind] + raw_DEG_synch[i]#/float(e*no_avg_itrs-raw_BER_synch[i]*N*no_avg_itrs)
            no_instances[ind] = no_instances[ind] + 1

        DEG_synch = np.divide(DEG_synch,no_instances)
        file_name = "./Plots/DEG_One_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_e_" + str(e0_for_deg) + "_X_" + str(max_decoding_itr)+".txt"    
        np.savetxt(file_name,np.vstack([itr_uniq,DEG_synch.T]).T,'%f',delimiter='\t',newline='\n')
        #-------------------------------------------------------------------------------

#===============================================================================

    
#=============================PLOT THE RESULTS==================================
e0_prob = np.divide(np.array(e0_uniq),float(N))
plt.grid(True)
plt.plot(e0_prob,BER_synch,'bs-',label='Standard BP',linewidth=3)
plt.plot(e0_prob,BER_asynch_non_uniform,'ro--',label='Asynchronous BP',linewidth=3)
plt.plot(e0_prob,BER_asynch_uniform,'g.-',label='Asynchronous BP, Jittered',linewidth=3)
plt.legend(framealpha=0.5,loc='south east')
plt.xlabel("$ \epsilon $")
plt.ylabel('BER')
plt.yscale('log')
plt.title('BER as a function of $\epsilon$ for $\Gamma$ = %s' %str(max_decoding_itr))
plt.show()


plt.grid(True)
plt.plot(e0_prob,TIME_synch,'bs-',label='Standard BP',linewidth=3)
plt.plot(e0_prob,TIME_asynch_non_uniform,'ro--',label='Asynchronous BP',linewidth=3)
plt.plot(e0_prob,TIME_asynch_uniform,'g.-',label='Asynchronous BP, Jittered',linewidth=3)
plt.legend(framealpha=0.5,loc='south west')
plt.xlabel("$ \epsilon $")
plt.ylabel('$t$')
#plt.yscale('log')
plt.title('Decoder time as a function of $\epsilon$ for $ \Gamma $ = %s' %str(max_decoding_itr))
plt.show()

plt.plot(e0_prob,ITR_synch,'bs-',label='Standard BP',linewidth=3)
plt.plot(e0_prob,ITR_asynch_non_uniform,'ro--',label='Asynchronous BP',linewidth=3)
plt.plot(e0_prob,ITR_asynch_uniform,'g.-',label='Asynchronous BP, Jittered',linewidth=3)
plt.legend(framealpha=0.5,loc='south west')
plt.xlabel("$ \epsilon $")
plt.ylabel('$t$')
plt.title('Number of decoding iterations as a function of $\epsilon$ for $ \Gamma $ = %s' %str(max_decoding_itr))
plt.show()


if track_deg_one_flag:
    plt.plot(itr_uniq,DEG_synch)
    plt.plot(itr_uniq,DEG_asynch_non_uniform,'r')
    plt.plot(itr_uniq,DEG_asynch_uniform,'g')
    plt.show()
#===============================================================================
