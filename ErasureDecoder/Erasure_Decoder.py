
#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import os
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
import random
from auxiliary_functions import *
#==============================================================================


#================================INITIALIZATIONS===============================
N = 128                                             # Codeword length
K = 64                                              # Messageword lengt
max_e = 20                                          # Maximum number of erasures to consider
min_e = 16                                          # Minimum number of erasures to consider
step_e = 8                                          # Erasure step

d_v = 4                                             # Degree of variable nodes
d_c = 8                                             # Degree of check nodes
d_max = 1                                           # Maximum (Mean in case of exponential distribution) delay of the edges in the parity check matrix. A value of 0 represents standard belief propagation
no_avg_itrs = 40                                    # Number of times a random noisy vector is generated for decoding
max_decoding_itr = 10000                              # Maximum number of iterations that the decodng algorithm is run after which a failure is declared
ensemble_size = 1                                   # The number of random graphs that will be considered in the simulations

track_deg_one_flag = 0                              # If equal to 1, the code tracks the number of degree one check nodes in the residual graph.
    
if not os.path.isdir('./Results'):                  # Create a folder if not already exists
    os.makedirs('./Results')
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hN:K:T:D:C:E:M:I:S:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-N':
            N = int(arg)                        # Number of variable nodes
        elif opt == '-K':
            K = int(arg)                        # Number of message nodes
        elif opt == '-T':
            no_avg_itrs = int(arg)              # Number of times a random noisy vector is generated for decoding
        elif opt == '-D':
            d_v = int(arg)                      # Degree of variable nodes
        elif opt == '-C':
            d_c = int(arg)                      # Degree of check nodes        
        elif opt == '-E':
            ensemble_size = int(arg)            # The number of random graphs that will be considered in the simulations        
        elif opt == '-M':
            max_e = int(arg)                    # Maximum number of erasures to consider
        elif opt == '-I':
            min_e = int(arg)                    # Minimum number of erasures to consider
        elif opt == '-S':
            step_e = int(arg)                   # Erasure step
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
    

e0_range = range(min_e,max_e,step_e)                        # Number of erased bits
#==============================================================================


for ensemble_itr in range(0,ensemble_size):

#==========================CREATE PARITY CHECK GRAPH===========================
    H,D_0,E0 = bipartite_graph(N,N-K,d_v,d_c,1,np.zeros([2,2]))
    H,D_orig,E = bipartite_graph(N,N-K,d_v,d_c,d_max,H)
    D_0 = np.sign(D_0) #D_0 * round(sum(sum(D))/float(sum(sum(D>0))))    
    
    final_BER_wo_Delay = np.zeros([1,len(e0_range)])    # The average bit error rate at the end of decoding for the delay-less decoding
    final_BER_synchronous = np.zeros([1,len(e0_range)]) # The average bit error rate at the end of decoding for synchronous decoding
    final_BER_w_Delay = np.zeros([1,len(e0_range)])                     # The average bit error rate at the end of decoding for the delayed decoding
    
    final_PER_w_Delay = np.zeros([1,len(e0_range)])                     # The average block error rate at the end of decoding for the delayed decoding
    final_PER_wo_Delay = np.zeros([1,len(e0_range)])    # The average block error rate at the end of decoding for the delay-less decoding
    final_PER_synchronous = np.zeros([1,len(e0_range)]) # The average block error rate at the end of decoding for synchronous decoding
    
    decoding_itr_w_Delay = np.zeros([1,len(e0_range)])                  # The average number of decoding iterations for synchronous decoding
    decoding_itr_wo_Delay = np.zeros([1,len(e0_range)])     # The average number of decoding iterations for the delay-less decoding
    decoding_itr_synchronous = np.zeros([1,len(e0_range)])  # The average number of decoding iterations for the delayed decoding
    
    decoding_time_wo_Delay = np.zeros([1,len(e0_range)])    # The average decoding time for synchronous decoding with uniform mean-delays    
    decoding_time_synchronous = np.zeros([1,len(e0_range)])  # The average decoding time for the synchrnous decoder
    decoding_time_w_Delay = np.zeros([1,len(e0_range)])                 # The average decoding time for synchronous decoding
#==============================================================================



#===========================PERFORM ERROR DECORDING============================
    itr_error = 0
    for e0 in e0_range:        
        
        for itr in range(0,no_avg_itrs):
    
            #-------------------Generate the Noisy Input Vector------------------------
            ind_e = range(0,N)
            random.shuffle(ind_e)
            x_init = np.zeros([1,N])            # The all-zero codeword is assumed
            x_init[0,ind_e[0:e0]] = -1         # The noisy codeword
            #-------------------------------------------------------------------------

            #=========================ASYNCHRONOUS DECODING===========================

            #------------------------Asynchronous Decoding-----------------------------
            err,decoding_itr,decoding_time,deg_one_w_delay_vs_itr = asynchronous_decoder(H,D_0,D_orig,E,x_init,max_decoding_itr,N,K,track_deg_one_flag,0,d_max)
            #-------------------------------------------------------------------------
                        
            final_BER_w_Delay[0,itr_error] = final_BER_w_Delay[0,itr_error] + err
            final_PER_w_Delay[0,itr_error] = final_PER_w_Delay[0,itr_error] + np.sign(err)
            decoding_itr_w_Delay[0,itr_error] = decoding_itr_w_Delay[0,itr_error] + decoding_itr
            decoding_time_w_Delay[0,itr_error] = decoding_time_w_Delay[0,itr_error] + decoding_time
            #--------------------------------------------------------------------------    

            #=====================JITTERED ASYNCHRONOUS DECODING=======================
            
            #------------------------Asynchronous Decoding-----------------------------
            err,decoding_itr,decoding_time,deg_one_wo_delay_vs_itr = asynchronous_decoder(H,D_0,D_orig,E,x_init,max_decoding_itr,N,K,track_deg_one_flag,1,d_max)
            #-------------------------------------------------------------------------

            #------------------------------Calculate BER-------------------------------
            final_BER_wo_Delay[0,itr_error] = final_BER_wo_Delay[0,itr_error] + err
            final_PER_wo_Delay[0,itr_error] = final_PER_wo_Delay[0,itr_error] + np.sign(err)
            decoding_itr_wo_Delay[0,itr_error] = decoding_itr_wo_Delay[0,itr_error] + decoding_itr
            decoding_time_wo_Delay[0,itr_error] = decoding_time_wo_Delay[0,itr_error] + decoding_time
            #--------------------------------------------------------------------------
        
            #==========================================================================
            
            
            #============================SYNCHRONOUS DECODING==========================
            
            #------------------------Asynchronous Decoding-----------------------------
            err,decoding_itr,decoding_time,deg_one_sync_vs_itr = synchronous_decoder(H,x_init,E,N,K,max_decoding_itr,track_deg_one_flag)
            #-------------------------------------------------------------------------

            #------------------------------Calculate BER-------------------------------
            final_BER_synchronous[0,itr_error] = final_BER_synchronous[0,itr_error] + err
            final_PER_synchronous[0,itr_error] = final_PER_synchronous[0,itr_error] + np.sign(err)
            decoding_itr_synchronous[0,itr_error] = decoding_itr_synchronous[0,itr_error] + decoding_itr
            decoding_time_synchronous[0,itr_error] = decoding_time_synchronous[0,itr_error] + decoding_time
            #--------------------------------------------------------------------------
        
            #==========================================================================
            
        #===============================SAVE THE RESULTS===============================
        file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_BER_w_Delay[0,itr_error]/float(N*no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_BER_wo_Delay[0,itr_error]/float(N*no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/BER_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_BER_synchronous[0,itr_error]/float(N*no_avg_itrs)))
        acc_file.close()
                
        file_name = "./Results/PER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_PER_w_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/PER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_PER_wo_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/PER_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_PER_synchronous[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
                
        file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_itr_w_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_itr_wo_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/ITR_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c)  + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_itr_synchronous[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
                
        file_name = "./Results/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_time_w_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/TIME_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_time_wo_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/TIME_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c)  + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_time_synchronous[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        
        #file_name = "./Results/DEG_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_e_" + str(e0) + ".txt"
        #np.savetxt(file_name,deg_w_delay_vs_itr/float(no_avg_itrs),'%3.5f',delimiter='\t',newline='\n')
        
        #file_name = "./Results/DEG_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + "_e_" + str(e0) + ".txt"
        #np.savetxt(file_name,deg_wo_delay_vs_itr/float(no_avg_itrs),'%3.5f',delimiter='\t',newline='\n')        
        
        #file_name = "./Results/DEG_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_e_" + str(e0) + ".txt"    
        #np.savetxt(file_name,deg_sync_vs_itr/float(no_avg_itrs),'%3.5f',delimiter='\t',newline='\n')
        
        
        if track_deg_one_flag:
            file_name = "./Results/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + "_e_" + str(e0) + ".txt"
            acc_file = open(file_name,'a')                        
            for i in range(0,100):
                acc_file.write("%d \t %f \n" %(i,deg_one_w_delay_vs_itr[i]/float(no_avg_itrs)))
            acc_file.close()
        
            file_name = "./Results/DEG_One_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + "_e_" + str(e0) + ".txt"
            acc_file = open(file_name,'a')                        
            for i in range(0,100):
                acc_file.write("%d \t %f \n" %(i,deg_one_wo_delay_vs_itr[i]/float(no_avg_itrs)))
            acc_file.close()
        
            file_name = "./Results/DEG_One_Sync_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_e_" + str(e0) + ".txt"    
            acc_file = open(file_name,'a')                        
            for i in range(0,100):
                acc_file.write("%d \t %f \n" %(i,deg_one_sync_vs_itr[i]/float(no_avg_itrs)))
            acc_file.close()
        #==============================================================================
        
        #----------------------------Print Some Results----------------------------
        print_str = 'No. input errors: '+ str(e0) +', Input BER: '+ str(e0/float(N)) + ', Outpur BER: '
        print_str = print_str + str(final_BER_w_Delay[0,itr_error]/float(N*no_avg_itrs)), ', w/o delay: ' + str(final_BER_wo_Delay[0,itr_error]/float(N*no_avg_itrs)) + ', sync: ' + str(final_BER_synchronous[0,itr_error]/float(N*no_avg_itrs))
        print print_str
        
        print_str = 'Avg. decoding itrs: with delay = ' + str(decoding_itr_w_Delay[0,itr_error]) 
        print_str = print_str + ', without delay = ' + str(decoding_itr_wo_Delay[0,itr_error]) + ', sync: ' + str(decoding_itr_synchronous[0,itr_error])
        print print_str
        
        print_str = 'Avg. decoding time: with delay = ' + str(decoding_time_w_Delay[0,itr_error]) 
        print_str = print_str + ', without delay = ' + str(decoding_time_wo_Delay[0,itr_error]) + ', sync: ' + str(decoding_time_synchronous[0,itr_error])
        print print_str
        #--------------------------------------------------------------------------
        
        itr_error = itr_error + 1
    
    
#==============================================================================
