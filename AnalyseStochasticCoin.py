import numpy as np
import pandas as pd
from scipy import optimize
from AnalyseStochasticCoin import *  

def OptimizeWithSeveralStartPostions(method,grid_start,grid_size,num_neg,mean_num_det_neg,counts_num_det_pos):
    
    a_ini=np.arange(grid_start,1,grid_size)    
    b_ini=np.arange(grid_start,1,grid_size)    
    
    optima=[]
    optima_val=[]

    #to be optimised by paralellisation or zipping 
    for a in a_ini:  
        
        foo=[optimize.minimize(Log_Liklihood_Coin_Toss_Stochastic,args=(num_neg,mean_num_det_neg,counts_num_det_pos), x0=[a, b],method=method,bounds=((0,1),(0,1)),jac=Score_Coin_Toss_Stochastic)
            for b in b_ini]

        LogL=[i.fun for i in foo]
        minpos = LogL.index(min(LogL))
        optima.append(foo[minpos].x)
        optima_val.append(min(LogL))
                          
                          
    minpos = optima_val.index(min(optima_val))
                     
    return optima[minpos]   

def Score_Coin_Toss_Stochastic(x,num_neg,mean_num_det_neg,counts_num_det_pos):

    ## Matrix is aa, ab, ba, bb
    
    # coint toss with proability a b^n for negative  
    if(x[1] < 1): 
        if(x[1] > 0):
            if(x[0] < 1):
                if(x[0] > 0): # no edge case
                    
                    foo1=0
                    foo2=0
                    
                    for i,num in enumerate(counts_num_det_pos):

                        if(num):
                            
                            fac=num *   np.power(x[1],i)/ (1-x[0] * np.power(x[1],i))
                            
                            foo1+= fac 
                            if(i):
                                foo2+=  i * fac

                    return -1 * np.array([num_neg/x[0] - foo1, num_neg * mean_num_det_neg/x[1] - x[0]/x[1] * foo2])

                else: # a  = 0, b no edge case
                    
                    if(num_neg):
                        return -[np.Inf,num_neg * mean_num_det_neg/x[1]]
                    else:
                        foo=0
                        for i,num in enumerate(counts_num_det_pos):

                            if(num):
                                foo=num *   np.power(x[1],i)
                            
                        return -1 * np.array([-foo,0])

            else: # a = 1, b no edge case
                foo1=0
                foo2=0

                for i,num in enumerate(counts_num_det_pos):

                    if(num):                        

                        fac=num *   np.power(x[1],i)/ ((1-np.power(x[1],i)))
                            
                        foo1+= fac 
                        if(i):
                            foo2+=  i * fac
                
                return -1 * np.array([num_neg/x[0] - foo1, num_neg * mean_num_det_neg/x[1] - 1/x[1] * foo2])
                
        else: # b = 0
            if(x[0] < 1): 
                if(x[0] > 0):
                    
                    if(counts_num_det_pos[0]):                    
                        return -1 * np.array([num_neg/x[0]-counts_num_det_pos[0]/(1-x[0]),-np.Inf])

                    else:
                        
                        if(num_neg):
                            if(mean_num_det_neg): # at least one negative drop coontains a cell

                                return -1 * np.array([num_neg/x[0],-np.Inf])
                            else:
                                return -1 * np.array([num_neg/x[0],0])

                        else:
                            return -1 * np.array([num_neg/x[0],0])

                else: # a  = 0   
                    
                    if(num_neg):
                        
                        if(mean_num_det_neg):
                            return -1 * np.array([np.Inf,np.Inf])

                        else:
                            return -1 * np.array([np.Inf,0])

                    else:
                        return np.array([0,0])

               
            else: # a = 1      
            
                if(counts_num_det_pos[0]): 
                    
                    if(num_neg):
                        
                        if(mean_num_det_neg):
                            
                            return -1 * np.array([-np.Inf,np.nan])
                        else: 
                            return -1 * np.array([np.Inf,np.Inf])

                    else:      
                    
                        return -1 * np.array([np.Inf,np.Inf])
                    
                else:
                    
                    if(num_neg):
                        
                        if(mean_num_det_neg):
                            return -1 * np.array([num_neg,np.Inf])
                        else: 
                            return -1 * np.array([num_neg,0])

                    else:
                        return np.array([0,0])

    else: # b = 1
        if(x[0] < 1): 
            if(x[0] > 0):
                return -1 * np.array([num_neg/x[0]- counts_num_det_pos.sum()/(1-x[0]),num_neg * mean_num_det_neg - x[0] * counts_num_det_pos.sum()/(1-x[0])])
            else: # a  = 0   
                
                if(num_neg):
                    return -1 * np.array([np.Inf,num_neg * mean_num_det_neg ])
                else:
                    return -1 * np.array([counts_num_det_pos.sum(),0])


        else: # a = 1  
            
            if(np.all(counts_num_det_pos == 0)):
                return -1 * num_neg * np.array([1,mean_num_det_neg])
            else:
                return -1 * np.array([np.Inf,np.Inf])


def Observed_Fisher_Information(x,num_neg,mean_num_det_neg,counts_num_det_pos):

    ## We return aa,ab,bb
    # coint toss with proability a b^n for negative  
    if(x[1] < 1): 
        if(x[1] > 0):
            if(x[0] < 1):
                if(x[0] > 0): # no edge case
                    
                    foo1=0
                    foo2=0
                    foo3=0

                    for i,num in enumerate(counts_num_det_pos):
                        if(num):
                            fac=counts_num_det_pos[i] / ((1- x[0] * np.power(x[1],i))**2)
                            foo1+=fac * np.power(x[1],2 * i)
                            if(i):
                                foo2+= fac * i *  np.power(x[1],i-1)
                                foo3+= fac * (i**2) *  np.power(x[1],i-2)

                    return [num_neg/x[0]**2+ foo1,foo2, x[0] * foo3]
                else: # a  = 0, b no edge case
                        
                    foo1=0
                    foo2=0

                    for i,num in enumerate(counts_num_det_pos):
                        if(num):
                            fac=counts_num_det_pos[i] 
                            foo1+=fac * np.power(x[1],2 * i)
                            if(i):
                                foo2+= fac * i *  np.power(x[1],i-1)

                    if(num_neg):            
                        return [np.Inf,foo2,0]

                    else:
                        return [foo1,foo2,0]

            
            else: # a = 1, b no edge case
                
                foo1=0
                foo2=0
                foo3=0

                for i,num in enumerate(counts_num_det_pos):
                    if(num):
                        fac=counts_num_det_pos[i] / ((1- np.power(x[1],i))**2)
                        foo1+=fac * np.power(x[1],2 * i)
                        if(i):
                            foo2+= fac * i *  np.power(x[1],i-1)
                            foo3+= fac * (i**2) *  np.power(x[1],i-2)

                return [num_neg+ foo1,foo2,foo3]
                
        else: # b = 0
            if(x[0] < 1): 
                if(x[0] > 0):
                    
                    if(counts_num_det_pos[1]):
                        return [num_neg/(x[0]**2) + counts_num_det_pos[0]/((1-x[0])**2),counts_num_det_pos[1]/((1-x[0])**2),np.Inf]

                    else:   
                        return [num_neg/(x[0]**2) + counts_num_det_pos[0]/((1-x[0])**2),0,counts_num_det_pos[2]/((1-x[0])**2)]
                else: # a  = 0   
                    
                    if(counts_num_det_pos[1]):
                        
                        if(num_neg):
                            return [np.Inf,counts_num_det_pos[1],np.Inf]
                        else:
                            return [counts_num_det_pos[0],counts_num_det_pos[1],np.Inf]

                    else:
                        if(num_neg):
                                return [np.Inf,counts_num_det_pos[1],counts_num_det_pos[2]]

                        else:
                                return [counts_num_det_pos[0],counts_num_det_pos[1],counts_num_det_pos[2]]

               
            else: # a = 1      
                if(counts_num_det_pos[1]):

                    if(counts_num_det_pos[0]):
                        return [np.Inf, np.Inf,np.Inf]
                    else:
                        return [num_neg, np.Inf,np.Inf]
                else:
                    if(counts_num_det_pos[0]):
                        
                        if(counts_num_det_pos[2]):
                            return [np.Inf, np.Inf,np.Inf]

                        else:  
                            return [np.Inf, np.Inf,0]
                    else:
                        
                        if(counts_num_det_pos[2]):
                            return [num_neg, np.Inf,np.Inf]

                        else:  
                            return [num_neg, np.Inf,0]
                    


    else: # b = 1
        if(x[0] < 1): 
            if(x[0] > 0):
                foo1=0
                foo2=0
                foo3=0

                for i,num in enumerate(counts_num_det_pos):
                    if(num):
                        fac=counts_num_det_pos[i] 
                        foo1+=fac
                        if(i):
                            foo2+= fac * i
                            foo3+= fac * (i**2) 

                return [num_neg/(x[0]**2)+ foo1,foo2,x[0] * foo3]
            else: # a  = 0   
                foo1=0
                foo2=0

                for i,num in enumerate(counts_num_det_pos):
                    if(num):
                        fac=counts_num_det_pos[i] 
                        foo1+=fac
                        if(i):
                            foo2+= fac * i

            
                if(num_neg):
                    return [np.Inf,foo2,0]

                else:
            
                    return [foo1,foo2,0]

        else: # a = 1  
            if(counts_num_det_pos): 
            
                if(np.all(counts_num_det_pos[1:] == 0)):

                    if(counts_num_det_pos[0]):

                        return [np.Inf,0,0]

                    else: # all are zero
                        return [num_neg,0,0]

                else: 
                    if(counts_num_det_pos[0]):# all are non zero

                        return [np.Inf,np.Inf,np.Inf]

                    else: # just zero is zero

                        return [num_neg,np.Inf,np.Inf]
            else:
                return [num_neg,0,0]

def Log_Liklihood_Coin_Toss_Stochastic(x,num_neg,mean_num_det_neg,counts_num_det_pos):
    
    # coint toss with proability a b^n for negative  
    if(x[1] < 1): 
        if(x[1] > 0):
            if(x[0] < 1):
                if(x[0] > 0): # no edge case
                    
                    foo=0
                    for i,num in enumerate(counts_num_det_pos):
                        if(num):
                            
                            foo2=1-x[0] * np.power(x[1],i)
                            if(foo2 >0):
                                foo+=  num * np.log(foo2)
                            else:
                                foo=np.NINF
                                break


                    return -(num_neg *(np.log(x[0])   + mean_num_det_neg * np.log(x[1])) + foo)
    
                    
                else: # a  = 0, b no edge case
                    if(num_neg):
                        return np.Inf
                    else:
                        return 0
                    
                    
            else: # a = 1, b no edge case

                foo=0
                for i,num in enumerate(counts_num_det_pos):
                    if(num):

                        foo2=1- np.power(x[1],i)
                        if(foo2 >0):
                            foo+=  num * np.log(foo2)
                        else:
                            foo=np.NINF
                            break


                return -(num_neg * mean_num_det_neg * np.log(x[1]) + foo)
            
            
        else: # b = 0
            if(num_neg):  # there are negative droplets
                if(mean_num_det_neg): # at least one negative drop coontains a cell
                    return np.Inf
                else: 
                    
                    if(x[0] < 1):
                        if(x[0]>0):
                            return -(num_neg * np.log(x[0])  + counts_num_det_pos[0] * np.log(1-x[0]))
                        else:  # a = 0
                            return np.Inf   
                        
                    else: # a = 1
                        if(counts_num_det_pos[0]): 
                            return  np.Inf
                        else:
                            return 0   
                    
                    
            else: # all positive drops
                
                if(x[0] < 1):
                    if(x[0]>0):
                        return -(counts_num_det_pos[0] * np.log(1-x[0]))
                    else:  # a = 0
                        return 0   
                        
                else: # a = 1
                    if(counts_num_det_pos[0]): 
                        return  np.Inf
                    else:
                        return 0   
                    
    else: # b = 1

        if(x[0] < 1): 
            if(x[0] > 0):
                    return  -(num_neg * np.log(x[0]) +    np.log(1-x[0]) * counts_num_det_pos.sum())
                
            else: # a = 0
                
                if(num_neg): 
                    return  np.Inf
                else:
                    return 0      
                
        else: # a = 1
    
            if(np.all(counts_num_det_pos == 0)):
                return 0      
            else:
                return  np.Inf                