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


#=======================DEFINING THE RELATED FUNCTIONS=========================

#-------------------Parity Check Graph Creation Function-----------------------
def bipartite_graph(N,M,d_v,d_c,d_max,H):
    #.................Sanity-Checking the Input Variables......................
    if (N*d_v != M*d_c):
        print 'Error! the graph specifications do not satisfy the required constraints'
        return
    else:
        E = int(N*d_v)
        if not sum(sum(abs(H))):
            H = np.zeros([M,N])
            graph_flag = 0
        else:
            graph_flag = 1
        D = np.zeros([M,N])
        E_dict = {}
    #..........................................................................
    
    #....................Create a Random Bipartite Graph.......................
    if not graph_flag:
        edges_slot = range(0,E)
        random.shuffle(edges_slot)
        var_index = 0
        check_index = 0
        for i in range(0,N):
            edge_ind = np.divide(edges_slot[i*d_v:(i+1)*d_v],d_c)
            
            H[edge_ind,i] = 1
            for jj in edge_ind:            
                E_dict['c_'+str(jj)+'_to_v_'+str(i)] = float('nan')
                E_dict['v_'+str(i)+'_to_c_'+str(jj)] = float('nan')
            D[edge_ind,i] = 1 + np.round(d_max*np.random.rand(d_v))        
    #..........................................................................
    
    #.............If the Graph is Given, Just Assign the Delyas................
    else:
        for i in range(0,N):
            for j in range(0,M):
                if (H[j,i]):
                    D[j,i] = 1 + np.round(d_max*np.random.rand(1))
                    E_dict['c_'+str(j)+'_to_v_'+str(i)] = float('nan')
                    E_dict['v_'+str(i)+'_to_c_'+str(j)] = float('nan')
    #..........................................................................
    
    return H,D,E_dict
    
#------------------------------------------------------------------------------

#------------------Variable-to-Check Message Transmission----------------------
def variable_to_check(v,c,m,E,H,D,t,queue_inds,event_queue):
        
    #.......................Shift Variable Messages............................
    M,N = H.shape    
    ind = np.nonzero(H[c,:])
    E['v_'+str(v)+'_to_c_'+str(c)] = m
    ind = list(ind[0])
    ind.remove(v)                  # Remove the variable node from the neighbors
    for j in ind:
        #~~~~~~~~~Update the Outgoing Messages to the Other Neighbors~~~~~~~~~~                
        temp_ind = ind
        temp_ind.remove(j)
        messages = []
        nan_flag = 0
        for jj in temp_ind:
            if np.isnan(E['v_'+str(jj)+'_to_c_'+str(c)]):
                nan_flag = 1
                break
            else:
                messages.append(E['v_'+str(jj)+'_to_c_'+str(c)])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~Add to Queue~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not nan_flag:
            
            if -1 in messages:
                s = -1
            else:
                s = np.mod(sum(messages),2)
            q_ind = find_queue_inds(queue_inds,t+D[c,j])                
            queue_inds.insert(q_ind,t+D[c,j])
            event_queue.insert(q_ind,[j,c,'v',s])                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    
    return E,queue_inds,event_queue
#------------------------------------------------------------------------------

#------------------Check-to-Variable Message Transmission----------------------
def check_to_variable(v,c,m,E,H,D,t,queue_inds,event_queue,var_states):
    
    #.......................Shift Variable Messages............................
    M,N = H.shape    
    ind = np.nonzero(H[:,v])
    E['c_'+str(c)+'_to_v_'+str(v)] = m
    
    if (m != -1) and (m != float('nan')):
        var_states[0,v] = m
        
        ind = list(ind[0])
        ind.remove(c)                  # Remove the variable node from the neighbors
        for j in ind:
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~Add to Queue~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                        
            q_ind = find_queue_inds(queue_inds,t+D[j,v])                
            queue_inds.insert(q_ind,t+D[j,v])
            event_queue.insert(q_ind,[v,j,'c',m])                
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    
    return E,queue_inds,event_queue,var_states
    
#------------------------------------------------------------------------------

#---------Find the Corresponding Variable Message to a Zero Checksum-----------
def find_var_message(decod_itr,var_messages,D):
    #.........................Shift Var Messages...............................        
    M,N = D.shape
    v = {}
    for i in range(0,M):
        ind = np.nonzero(D[i,:])
        ind = ind[0]        
        for j in ind:
            if str(j) in v:
                v[str(j)].append(var_messages[decod_itr-D[i,j],j])
            else:
                v[str(j)] = [var_messages[decod_itr-D[i,j],j]]                
    #..........................................................................
    
    #........................Take the Majority.................................
    x = float('nan')*np.ones([1,N])
    for j in range(0,N):
        temp = v[str(j)]
        if sum(temp)>=(len(temp)/2.0):
            x[0,j] = 1
        else:
            x[0,j] = 0
    #..........................................................................
    
    return x
#------------------------------------------------------------------------------

#==============================================================================


#=========================PUSH TO QUEUE FUNCTIONS==============================
def find_queue_inds(queue_inds,t):
    
    if (len(queue_inds) == 0):
        ind = -1    
    elif (len(queue_inds) == 1):
        if t > queue_inds[0]:
            ind = 1            
        else:
            ind = 0                        
    #................Recursiveley Find the Correct Positions...................
    else:
        m = len(queue_inds)/2
        if (t <= queue_inds[m]):
            ind = find_queue_inds(queue_inds[0:m],t)
        else:
            ind = m+1 + find_queue_inds(queue_inds[m:],t)
    #..........................................................................    

    return ind
#==============================================================================


#================================INITIALIZATIONS===============================
N = 128                                             # Codeword length
K = 64                                              # Messageword length
e0_range = range(0,int(N),4)                        # Number of erased bits
d_v = 4                                             # Degree of variable nodes
d_c = 8                                             # Degree of check nodes
d_max = 10                                          # Maximum delay of the edges in the parity check matrix. A value of 0 represents standard belief propagation
no_avg_itrs = 40                                    # Number of times a random noisy vector is generated for decoding
max_decoding_itr = 80000                            # Maximum number of iterations that the decodng algorithm is run after which a failure is declared
final_BER_w_Delay = np.zeros([1,len(e0_range)])     # The average bit error rate at the end of decoding for the delayed decoding
final_BER_wo_Delay = np.zeros([1,len(e0_range)])    # The average bit error rate at the end of decoding for the delay-less decoding
var_messages = float('nan') * np.ones([max_decoding_itr,N])       # The time-stamped messages sent by variable nodes
check_messages = float('nan') * np.ones([max_decoding_itr,N-K])   # The time-stamped messages sent by check nodes
decoding_itr_wo_Delay = np.zeros([1,len(e0_range)])     # The average number of decoding iterations for the delay-less decoding
decoding_itr_w_Delay = np.zeros([1,len(e0_range)])      # The average number of decoding iterations for the delayed decoding
ensemble_size = 1                                   # The number of random graphs that will be considered in the simulations
if not os.path.isdir('./Results'):                  # Create a folder if not already exists
    os.makedirs('./Results')
#==============================================================================


for ensemble_itr in range(0,ensemble_size):

#==========================CREATE PARIT CHECK GRAPH============================
    H,D_0,E0 = bipartite_graph(N,N-K,d_v,d_c,0,np.zeros([2,2]))
    H,D,E = bipartite_graph(N,N-K,d_v,d_c,d_max,H)
    D_0 = D_0 * round(sum(sum(D))/float(sum(sum(D>0))))
#==============================================================================



#=========================PERFORM ERROR DECORDING==============================
    itr_error = 0
    for e0 in e0_range:    

        for itr in range(0,no_avg_itrs):
    
            #-------------------Generate the Noisy Input Vector------------------------
            ind_e = range(0,N)
            random.shuffle(ind_e)
            x_init = np.zeros([1,N])            # The all-zero codeword is assumed
            x_init[0,ind_e[0:e0]] = -1         # The noisy codeword
            #--------------------------------------------------------------------------
    
            #----------------Fill Event Queue with Initial Variables-------------------
            event_queue = []
            queue_inds = []
            var_states = copy.deepcopy(x_init)
            for v in range(0,N):
                ind = np.nonzero(H[:,v])
                ind = ind[0]            
                for c in ind:
                    q_ind = find_queue_inds(queue_inds,D[c,v])
                    if q_ind < 0:
                        queue_inds = [D[c,v]]
                        event_queue.append([v,c,'c',x_init[0,v]])
                    else:
                        queue_inds.insert(q_ind,D[c,v])
                        event_queue.insert(q_ind,[v,c,'c',x_init[0,v]])                
            #--------------------------------------------------------------------------
        
            #-------------Process the Events in the Queue Until Finished---------------
            decoding_itr = 0
            while (len(event_queue) and (decoding_itr < max_decoding_itr) ):            
            
                #.................Pop The Top Event From the Queue.....................
                event_list = event_queue[0]
                t = queue_inds[0]
                del queue_inds[0]
                del event_queue[0]
                #......................................................................

                #.........................Process the Event............................
                v = event_list[0]
                c = event_list[1]
                m = event_list[3]
                if (event_list[2] == 'c'):                
                    E,queue_inds,event_queue = variable_to_check(v,c,m,E,H,D,t,queue_inds,event_queue)
                elif (event_list[2] == 'v'):
                    E,queue_inds,event_queue,var_states = check_to_variable(v,c,m,E,H,D,t,queue_inds,event_queue,var_states)
                else:
                    print 'Invalid event type!'                
                #......................................................................
            
                #......................Check Stopping Condition........................
                if (-1 not in var_states):
                    print 'Asyncronous decoding finished successfully in %d iterations' %decoding_itr
                    break
                else:
                    decoding_itr = decoding_itr + 1
                #......................................................................
    
            #--------------------------------------------------------------------------

            #------------------------------Calculate BER-------------------------------
            final_BER_w_Delay[0,itr_error] = final_BER_w_Delay[0,itr_error] + sum(abs(var_states[0,:]) > 0)
            decoding_itr_w_Delay[0,itr_error] = decoding_itr_w_Delay[0,itr_error] + decoding_itr
            #--------------------------------------------------------------------------

            #----------------Fill Event Queue with Initial Variables-------------------
            event_queue = []
            queue_inds = []
            var_states = copy.deepcopy(x_init)
            E = E0
        
            for v in range(0,N):
                ind = np.nonzero(H[:,v])
                ind = ind[0]            
                for c in ind:
                    q_ind = find_queue_inds(queue_inds,D_0[c,v])
                    if q_ind < 0:
                        queue_inds = [D[c,v]]
                        event_queue.append([v,c,'c',x_init[0,v]])
                    else:
                        queue_inds.insert(q_ind,D_0[c,v])
                        event_queue.insert(q_ind,[v,c,'c',x_init[0,v]])        
            #--------------------------------------------------------------------------
        
            #-------------Process the Events in the Queue Until Finished---------------
            decoding_itr = 0
            while (len(event_queue) and (decoding_itr < max_decoding_itr) ):            
            
                #.................Pop The Top Event From the Queue.....................
                event_list = event_queue[0]
                t = queue_inds[0]
                del queue_inds[0]
                del event_queue[0]
                #......................................................................

                #.........................Process the Event............................
                v = event_list[0]
                c = event_list[1]
                m = event_list[3]
                if (event_list[2] == 'c'):                
                    E,queue_inds,event_queue = variable_to_check(v,c,m,E,H,D_0,t,queue_inds,event_queue)
                elif (event_list[2] == 'v'):
                    E,queue_inds,event_queue,var_states = check_to_variable(v,c,m,E,H,D_0,t,queue_inds,event_queue,var_states)
                else:
                    print 'Invalid event type!'                
                #......................................................................
            
                #......................Check Stopping Condition........................
                if (-1 not in var_states):
                    print 'Decoding finished successfully in %d iterations' %decoding_itr
                    break
                else:
                    decoding_itr = decoding_itr + 1
                #......................................................................
    
            #--------------------------------------------------------------------------

            #------------------------------Calculate BER-------------------------------
            final_BER_wo_Delay[0,itr_error] = final_BER_wo_Delay[0,itr_error] + sum(abs(var_states[0,:]) > 0)
            decoding_itr_wo_Delay[0,itr_error] = decoding_itr_wo_Delay[0,itr_error] + decoding_itr
            #--------------------------------------------------------------------------
        
        file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_BER_w_Delay[0,itr_error]/float(N*no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/BER_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,final_BER_wo_Delay[0,itr_error]/float(N*no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(d_max) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_itr_w_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        file_name = "./Results/ITR_N_" + str(N) + "_K_" + str(K) + "_deg_" + str(d_v) + "_" + str(d_c) + "_D_" + str(0) + ".txt"
        acc_file = open(file_name,'a')                        
        acc_file.write("%d \t %f \n" %(e0,decoding_itr_wo_Delay[0,itr_error]/float(no_avg_itrs)))
        acc_file.close()
        
        #----------------------------Print Some Results----------------------------
        print_str = 'No. input errors: '+ str(e0) +', Input BER: '+ str(e0/float(N)) + ', Outpur BER: ' + str(final_BER_w_Delay[0,itr_error]), ', w/o delay: ' + str(final_BER_wo_Delay[0,itr_error])
        print print_str
        print_str = 'Avg. decoding itrs: with delay = ' + str(decoding_itr_w_Delay[0,itr_error]) + ', without delay = ' + str(decoding_itr_wo_Delay[0,itr_error])
        print print_str
        #--------------------------------------------------------------------------
        itr_error = itr_error + 1
    
    
#==============================================================================

final_BER_w_Delay = final_BER_w_Delay/float(N*no_avg_itrs*ensemble_size)
final_BER_wo_Delay = final_BER_wo_Delay/float(N*no_avg_itrs*ensemble_size)
decoding_itr_w_Delay = decoding_itr_w_Delay/float(no_avg_itrs*ensemble_size)
decoding_itr_wo_Delay = decoding_itr_wo_Delay/float(no_avg_itrs*ensemble_size)