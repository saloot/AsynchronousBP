
#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import os
import pdb
import random
import copy
#==============================================================================


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
            D[edge_ind,i] = np.random.exponential(d_max,[d_v]) #1 + np.round(d_max*np.random.rand(d_v))        
    #..........................................................................
    
    #.............If the Graph is Given, Just Assign the Delyas................
    else:
        for i in range(0,N):
            for j in range(0,M):
                if (H[j,i]):
                    D[j,i] =  np.random.exponential(d_max) #1 + np.round(d_max*np.random.rand(1))
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
        if np.isnan(m):
            nan_flag = 1
        elif m == -1:
            messages = [-1]
            nan_flag = 1
        else:
            messages = [m]      # double check
            nan_flag = 0
            for jj in temp_ind:
                if np.isnan(E['v_'+str(jj)+'_to_c_'+str(c)]):
                    nan_flag = 1
                    break
                elif (E['v_'+str(jj)+'_to_c_'+str(c)] == -1):
                    messages = [-1]
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
                
            if 1:#s>-1:                # Double check
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

#--------------Synchrnous Variable-to-Check Message Transmission---------------
def variable_to_check_sync(E,H):
        
    #.......................Shift Variable Messages............................
    M,N = H.shape    
    
    for i in range(0,M):
        ind = np.nonzero(H[i,:])
        ind = list(ind[0])
        for j in ind:
            #~~~~~~~~~Update the Outgoing Messages to the Other Neighbors~~~~~~~~~~                
            temp_ind = ind
            temp_ind.remove(j)
            messages = []
            nan_flag = 0
            for jj in temp_ind:        
                if np.isnan(E['v_'+str(jj)+'_to_c_'+str(i)]):
                    nan_flag = 1
                    break
                elif (E['v_'+str(jj)+'_to_c_'+str(i)]==-1):
                    messages = [-1]
                    break
                else:
                    messages.append(E['v_'+str(jj)+'_to_c_'+str(i)])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~Add to Queue~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not nan_flag:
            
                if -1 in messages:
                    s = -1
                else:
                    s = np.mod(sum(messages),2)
            
                for jj in temp_ind:
                    E['c_'+str(i)+'_to_v_'+str(j)] = s
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    return E
#------------------------------------------------------------------------------

#------------------Check-to-Variable Message Transmission----------------------
def check_to_variable_sync(E,H,var_states):
    
    #.......................Shift Variable Messages............................
    M,N = H.shape
    for j in range(0,N):
        ind = np.nonzero(H[:,j])
        ind = list(ind[0])
        for i in ind:
            #~~~~~~~~~Update the Outgoing Messages to the Other Neighbors~~~~~~~~~~                
            temp_ind = ind
            temp_ind.remove(i)
            messages = []
            nan_flag = 0
            for ii in temp_ind:                        
                messages.append(E['c_'+str(ii)+'_to_v_'+str(j)])
        
            if (0 in messages) or (1 in messages):
                
                m0 = sum(np.array(messages) == 0)
                m1 = sum(np.array(messages) == 1)
                #pdb.set_trace()
                if m1 >= m0:
                    m = 1
                    var_states[0,j] = m
                else:
                    m = 0
                    var_states[0,j] = m
            else:
                m = -1
        
            for ii in temp_ind:
                E['v_'+str(j)+'_to_c_'+str(i)] = m
        
    
    
    return E,var_states
    
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
            ind = m + find_queue_inds(queue_inds[m:],t)
    #..........................................................................    

    return ind
#==============================================================================


#==========================TRACK DEGREE ONE NODES==============================
def track_deg_one(H,var_states):
    M,N = H.shape
    v = copy.deepcopy(var_states)
    v = (v==-1).astype(int)    
    rho = np.dot(H,v.T)    
    deg_dist,bins = np.histogram(rho,range(0,M+1))
    return deg_dist
#==============================================================================


#========================THE ASYNCHRONOUS DECODER==============================
def asynchronous_decoder(H,D_0,D_orig,E,x_init,max_decoding_itr,N,K,track_deg_one_flag,jitter_flag,d_max):
        
    #-----------------------------Initialization-------------------------------    
    deg_one_w_delay_vs_itr = np.zeros([1000])                           # To track the degre one nodes
    var_messages = float('nan') * np.ones([max_decoding_itr,N])         # The time-stamped messages sent by variable nodes
    check_messages = float('nan') * np.ones([max_decoding_itr,N-K])     # The time-stamped messages sent by check nodes
    D = D_orig
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
    t_flag = 0
    while (len(event_queue) and (decoding_itr < max_decoding_itr) ):            
            
        #.....................Sample Delay Matrix at Random....................
        if jitter_flag:
            rand_D = D_orig + (0.1*(np.random.rand(N-K,N)-0.5))
        else:
            rand_D = np.random.exponential(d_max,[N-K,N])                
        D = np.multiply(D_0,rand_D)
        #......................................................................
                
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
            
        #.........Track the Degree Distribution on the Residual Graph..........
        if track_deg_one_flag:
            dd = track_deg_one(H,var_states)                
            deg_w_delay_vs_itr[decoding_itr,:] = deg_w_delay_vs_itr[decoding_itr,:] + dd
                    
            if (t>int(round(t))):
                if (t_flag == 0):
                    if int(round(t))<len(deg_one_w_delay_vs_itr)-1:
                        deg_one_w_delay_vs_itr[int(round(t))] = deg_one_w_delay_vs_itr[int(round(t))] + dd[1]                    
                        t_flag = 1                        
            else:                    
                t_flag = 0
        #......................................................................
                
        #......................Check Stopping Condition........................
        decoding_itr = decoding_itr + 1 #d_v + d_c
        if (-1 not in var_states):
            #print 'Asyncronous decoding finished successfully in %d iterations' %decoding_itr
            break
        #......................................................................
                
    #--------------------------------------------------------------------------

    #------------------------------Calculate BER-------------------------------            
    err = sum(abs(var_states[0,:]) > 0)
    return err,decoding_itr,t,deg_one_w_delay_vs_itr            
    #--------------------------------------------------------------------------
    
    #==============================================================================
    
    
#===========================THE YNCHRONOUS DECODER=================================
def synchronous_decoder(H,x_init,E,N,K,max_decoding_itr,track_deg_one_flag):

    #----------------Fill Event Queue with Initial Variables-------------------            
    var_states = copy.deepcopy(x_init)
    deg_one_sync_vs_itr = np.zeros([1000])          # To track the degre one nodes
    #--------------------------------------------------------------------------
            
    #-------------Process the Events in the Queue Until Finished---------------
    decoding_itr = 0
    t_flag = 0
    while (decoding_itr < max_decoding_itr):
            
        #.........................Process the Event............................
        E = variable_to_check_sync(E,H)
        var_states_old = copy.deepcopy(var_states)
        E,var_states = check_to_variable_sync(E,H,var_states)
        #......................................................................
            
                
        #.........Track the Degree Distribution on the Residual Graph..........
        if track_deg_one_flag:
            dd = track_deg_one(H,var_states)
            deg_sync_vs_itr[decoding_itr,:] = deg_sync_vs_itr[decoding_itr,:] + dd
            if int(decoding_itr/(2*len(E))) < len(deg_one_sync_vs_itr)-1:
                deg_one_sync_vs_itr[int(decoding_itr/(2*len(E)))] = deg_one_sync_vs_itr[int(decoding_itr/(2*len(E)))] + dd[1]
        #......................................................................
                
        #......................Check Stopping Condition........................
        decoding_itr = decoding_itr + 2*len(E)
        if (-1 not in var_states):                                        
            #print 'Synchronous decoding finished successfully in %d iterations' %decoding_itr
            break
                
        if sum(sum(abs(var_states - var_states_old))) == 0:
            break
        #......................................................................
    
    #--------------------------------------------------------------------------
    
    err = sum(abs(var_states[0,:]) > 0)
    t = decoding_itr/float(len(E))
    return err,decoding_itr,t,deg_one_sync_vs_itr

    #==============================================================================



    