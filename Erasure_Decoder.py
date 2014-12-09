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
            E_dict['c_'+str(edge_ind)+'_to_v_'+str(i)] = float('nan')
            E_dict['v_'+str(i)+'_to_c_'+str(edge_ind)] = float('nan')
            D[edge_ind,i] = 1 + np.round(d_max*np.random.rand(d_v))        
    #..........................................................................
    
    #.............If the Graph is Given, Just Assign the Delyas................
    else:
        for i in range(0,N):
            for j in range(0,M):
                if (H[j,i]):
                    D[j,i] = 1 + np.round(d_max*np.random.rand(1))
    #..........................................................................
    
    return H,D,E_dict
    
#------------------------------------------------------------------------------

#------------------Variable-to-Check Message Transmission----------------------
def variable_to_check(v,c,m,E,H,D,t,queue_inds,event_queue):
        
    #.......................Shift Variable Messages............................
    M,N = H.shape    
    ind = np.nonzero(H[v,:])
    E['v_'+str(i)+'_to_c_'+str(c)] = m
    ind = ind[0]
    ind.remove(v)                  # Remove the variable node from the neighbors
    for j in ind:
        #~~~~~~~~~Update the Outgoing Messages to the Other Neighbors~~~~~~~~~~                
        temp_ind = ind
        temp_ind.remove(j)
        messages = []
        nan_flag = 0
        for jj in temp_ind:
            if isnan(E['v_'+str(jj)+'_to_c_'+str(c)]):
                nan_flag = 1
                break
            else:
                messages.append(E['v_'+str(jj)+'_to_c_'+str(c)])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~Add to Queue~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not nan_flag:
            s = np.mod(sum(messages),2)
            q_ind = find_queue_inds(queue_inds,t+D[c,j])                
            queue_inds.insert(q_ind,D[c,j])
            event_queue.insert(q_ind,[j,c,'v',s])                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    
    return E,queue_inds,event_queue
#------------------------------------------------------------------------------

#------------------Check-to-Variable Message Transmission----------------------
def check_to_variable(x0,y_matrix,H,D,theta,t):
    #.........................Shift Check Messages.............................
    M,N = H.shape
    y = float('nan')*np.ones([1,M])
    for j in range(0,N):
        ind = np.nonzero(H[:,j])
        for i in ind[0]:
            d = D[i,j]
            if (t+1-d)>=0:
                y[0,i] = y_matrix[t+1-d,i]
    #..........................................................................
    
    #.......................Create a Check Messages............................
    v = np.ma.masked_array(y,mask= (y=='nan'))    
    x = np.ma.dot(v,H)
    x = x.data
    one_ind = (x>theta*sum(H)).astype(int)
    zero_ind = (x<(1-theta)*sum(H)).astype(int)    
    x = x0-one_ind - np.multiply(x0,zero_ind)# + np.multiply(1-one_ind,1-zero_ind)
    
    x = abs(x)
    update_flag = one_ind + zero_ind
    #..........................................................................
    
    return x
    
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
            ind = find_queue_inds(queue_inds[0:m+1],t)
        else:
            ind = m+1 + find_queue_inds(queue_inds[m+1:],t)
    #..........................................................................    

    return ind
#==============================================================================


#================================INITIALIZATIONS===============================
N = 128                                             # Codeword length
K = 64                                              # Messageword length
e0_range = range(0,int(N/8))                        # Number of erased bits
d_v = 4                                             # Degree of variable nodes
d_c = 8                                             # Degree of check nodes
d_max = 10                                          # Maximum delay of the edges in the parity check matrix. A value of 0 represents standard belief propagation
no_avg_itrs = 40                                    # Number of times a random noisy vector is generated for decoding
max_decoding_itr = 80                               # Maximum number of iterations that the decodng algorithm is run after which a failure is declared
final_BER_w_Delay = np.zeros([1,len(e0_range)])     # The average bit error rate at the end of decoding for the delayed decoding
final_BER_wo_Delay = np.zeros([1,len(e0_range)])    # The average bit error rate at the end of decoding for the delay-less decoding
var_messages = float('nan') * np.ones([max_decoding_itr,N])       # The time-stamped messages sent by variable nodes
check_messages = float('nan') * np.ones([max_decoding_itr,N-K])   # The time-stamped messages sent by check nodes

#==============================================================================

#==========================CREATE PARIT CHECK GRAPH============================
H,D_0,E = bipartite_graph(N,N-K,d_v,d_c,0,np.zeros([2,2]))
H,D,[] = bipartite_graph(N,N-K,d_v,d_c,d_max,H)
#==============================================================================



#=========================PERFORM ERROR DECORDING==============================
itr_error = 0
for e0 in e0_range:    

    for itr in range(0,no_avg_itrs):
    
        #-------------------Generate the Noisy Input Vector------------------------
        ind_e = range(0,N)
        random.shuffle(ind_e)
        x_init = np.zeros([1,N])            # The all-zero codeword is assumed
        x_init[0,ind_e[0:e0]] = 1       # The noisy codeword
        #--------------------------------------------------------------------------
    
        #----------------Fill Event Queue with Initial Variables-------------------
        event_queue = {}
        queue_inds = {}
        var_states = x_init
        for v in range(0,N):
            ind = np.nonzero(H[:,v])
            ind = ind[0]            
            for c in ind[0]:
                q_ind = find_queue_inds(queue_inds,D[c,v])
                if q_ind < 0:
                    queue_inds = [D[c,v]]
                else:
                    queue_inds.insert(q_ind,D[c,v])
                    event_queue.insert(q_ind,[v,c,'c',x_init[v]])
        #--------------------------------------------------------------------------
        
        #-------------Process the Events in the Queue Until Finished---------------
        while (len(event_queue)):
            
            #.................Pop The Top Event From the Queue.....................
            event_list = event_queue[0]
            t = queue_inds[0]
            del queue_inds[0]
            del event_queue[0]
            #......................................................................

            #.........................Process the Event............................
            if (event_list[2] == 'c'):
                v = event_list[0]
                c = event_list[1]
                m = event_list[3]
                E = variable_to_check(v,c,m,E)
            elif (event_list[2] == 'v'):
                
            else:
                print 'Invalid event type!'
                return            
            #......................................................................
            
    
        #--------------------------------------------------------------------------


        #------------------Apply Iterative Decoding With Delay---------------------
        var_messages.fill(float('nan'))
        check_messages.fill(float('nan'))
        var_messages[0,:] = x_init
        for decod_itr in range(1,max_decoding_itr):
            check_messages[decod_itr,:] = variable_to_check(var_messages,H,D,decod_itr)
            #if sum(check_messages[decod_itr,:]):
        
            x0 = var_messages[decod_itr-1,:]
            var_messages[decod_itr,:] = check_to_variable(x0,check_messages,H,D,0.8,decod_itr)
        
            #if (sum(sum(abs(check_messages[max(0,decod_itr-2*d_max):decod_itr+1,:]))) < 1e-8):
            if (sum(check_messages[decod_itr,:]) < 1e-8):
                break
        #--------------------------------------------------------------------------
    
        #------------------------------Calculate BER-------------------------------
        #v = find_var_message(decod_itr,var_messages,D)
        #final_BER_w_Delay[0,itr_error] = final_BER_w_Delay[0,itr_error] + sum(np.sign(v[0]))
        if (sum(check_messages[decod_itr,:])):
            final_BER_w_Delay[0,itr_error] = final_BER_w_Delay[0,itr_error] + N
        #--------------------------------------------------------------------------
        
        
        #-----------------Apply Iterative Decoding Without Delay-------------------
        var_messages.fill(float('nan'))
        check_messages.fill(float('nan'))
        var_messages[0,:] = x_init
        for decod_itr in range(1,max_decoding_itr):
            check_messages[decod_itr,:] = variable_to_check(var_messages,H,D_0,decod_itr)
            #if sum(check_messages[decod_itr,:]):
        
            x0 = var_messages[decod_itr-1,:]
            var_messages[decod_itr,:] = check_to_variable(x0,check_messages,H,D_0,0.6,decod_itr)
        
            if (sum(check_messages[decod_itr,:]) < 1e-8):
                break
        #--------------------------------------------------------------------------
    
        #------------------------------Calculate BER-------------------------------
        #v = find_var_message(decod_itr,var_messages,D_0)
        #final_BER_wo_Delay[0,itr_error] = final_BER_wo_Delay[0,itr_error] + sum(np.sign(v[0]))
        if (sum(check_messages[decod_itr,:])):
            final_BER_wo_Delay[0,itr_error] = final_BER_wo_Delay[0,itr_error] + N
        #--------------------------------------------------------------------------
        
    final_BER_w_Delay[0,itr_error] = final_BER_w_Delay[0,itr_error]/float(N*no_avg_itrs)
    final_BER_wo_Delay[0,itr_error] = final_BER_wo_Delay[0,itr_error]/float(N*no_avg_itrs)
    
    #----------------------------Print Some Results----------------------------
    print_str = 'No. input errors: '+ str(e0) +', Input BER: '+ str(e0/float(N)) + ', Outpur BER: ' + str(final_BER_w_Delay[0,itr_error]), ', w/o delay: ' + str(final_BER_wo_Delay[0,itr_error])
    print print_str
    
    #--------------------------------------------------------------------------
    itr_error = itr_error + 1
    
    
#==============================================================================

    