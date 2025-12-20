import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson,nbinom

import os
from scipy import optimize
from scipy.special import psi
from scipy.special import gamma, factorial

from functools import partial



def Liklihood_Neg_Binom(param,probs,mean,M):
    # param= lambda, r
    foo=np.array([np.log( gamma(param[1] + i)/(gamma(param[1]) * factorial(i)))  for i in range(len(probs)) ] )
    
    return M * (param[1] * np.log(1+param[0]/param[1]) + mean * np.log(1+param[1]/param[0]) - np.dot(foo, probs))
    
        
def Score_NegBinomial_r(n,r):
    
    # parametrised as mean value and r, estimate for mean value already plugged in
    if(r>0):
        
        foo=0
        for i in n:
            foo+=psi(r+i)

        N=len(n)

        return foo - N * psi(r) + N * (np.log(r) - np.log(r+ np.mean(n)))

    else:
        return -np.inf
    
    

def AnalyseInitialCount(data):

    out=pd.DataFrame({'lam': [data.mean()],'var': [data.var()]})
    out['r']=np.power(out['lam'],2)/(out['var']-out['lam'])
    out['var_lam']=   out['var'] - out['lam']
        
    f = partial(Score_NegBinomial_r,data)
    
    if(out['r'].values[0] > 0):
        opt=optimize.root(f,out['r'].values[0], method='krylov')
    else:
        opt=optimize.root(f,1, method='krylov',) 

    
    if(opt.success):
        out['r_like']=opt.x
    else:
        print('Root finding was not successful !')
        out['r-like']=np.nan
        
    # now Poisson propagated    
    counts=np.bincount(data.values)
    probs=counts/np.sum(counts)
    
    return out
