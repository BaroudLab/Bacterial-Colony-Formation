import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import nbinom
from scipy.special import psi, polygamma
from functools import partial
from AnalyseStochasticCoin import *  

def Estimate_Fisher_Information_NegBinom(x,lam_rho,r,num_samples):
      
        
    chip_1 = pd.DataFrame(np.random.negative_binomial(r,r/(lam_rho+r), num_samples),columns=['cells'])
    chip_2 = pd.DataFrame(np.random.negative_binomial(r,r/(lam_rho+r), num_samples),columns=['cells'])
    chip_3 = pd.DataFrame(np.random.negative_binomial(r,r/(lam_rho+r), num_samples),columns=['cells'])     

    chip_1['aa_estimate']=chip_1['cells'].apply(lambda n: np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )
    chip_2['ab_estimate']=chip_2['cells'].apply(lambda n: n*  np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )
    chip_3['bb_estimate']=chip_3['cells'].apply(lambda n: n**2 * np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )
    
    return [ chip_1['aa_estimate'].mean()/x[0], chip_2['ab_estimate'].mean()/x[1],(x[0]/x[1]**2) * chip_3['bb_estimate'].mean()]   


def Score_NegBinomial_r(n,r):
    
    # parametrised as mean value and r, estimate for mean value already plugged in

    foo=0
    for i in n:
        foo+=psi(r+i)

    N=len(n)
    return foo - N * psi(r) + N * (np.log(r) - np.log(r+ np.mean(n)))

def Estimate_Fisher_Information_part_r(lam_rho,r,num_samples):
    
       
    chip_1 = pd.DataFrame(np.random.negative_binomial(r,r/(lam_rho+r), num_samples),columns=['cells'])

    chip_1['estimate']=chip_1['cells'].apply(lambda n:  polygamma(1,r+n))
    
    return polygamma(1,r)-lam_rho/r * 1/(r+lam_rho) - chip_1['estimate'].mean()


def Log_Liklihood_Coin_Toss_Stochastic_NegBinom(x,r,num_neg,mean_num_det_neg,counts_num_det_pos):

    #input is Phi and q
    
    if(x[1]<1e-10): #essentially zero
        b= 0
        a= np.power(1/(1 + x[0]),r)
    else:
        b= x[1] * 1/(1 + x[0] * (1-x[1]))
        a= np.power(b/x[1],r)
    
    return  Log_Liklihood_Coin_Toss_Stochastic([a,b],num_neg,mean_num_det_neg,counts_num_det_pos)
 

    
def  Score_Coin_Toss_Stochastic_NegBinom(x,r,num_neg,mean_num_det_neg,counts_num_det_pos):
    
    #input is Phi and q
    
    if(x[1]<1e-10): #essentially zero
        b= 0
        a= np.power(1/(1 + x[0]),r)
    else:
        b= x[1] * 1/(1 + x[0] * (1-x[1]))
        a= np.power(b/x[1],r)
    
    score_ab=Score_Coin_Toss_Stochastic([a,b],num_neg,mean_num_det_neg,counts_num_det_pos)    
                      
    foo = 1/(1 + x[0] * (1-x[1]))
                      
    B= -r * a * (1-x[1]) * foo
    C=- x[1] * (1-x[1])  * (foo**2)
    D = r * a * x[0] * foo
    E= (1+x[0]) * (foo**2)
    
    
    return np.array([B * score_ab[0] + C * score_ab[1], D * score_ab[0] + E * score_ab[1]] )

    
def Fisher_Information_NegBinom(x,lam_rho,r,num_samples):
           
    #input is Phi and q
    if(x[1]<1e-10): #essentially zero
        b= 0
        a= np.power(1/(1 + x[0]),r)
    else:
        b= x[1] * 1/(1 + x[0] * (1-x[1]))
        a= np.power(b/x[1],r)
        
        
    FI_ab =Estimate_Fisher_Information_NegBinom([a,b],lam_rho,r,num_samples)

    foo = 1/(1 + x[0] * (1-x[1]))
    
    
    # all FI divergent if a=b=1 and hence here only if q=1
    # aa divergent for a =1, and hence here for q=1 (already covered) and lambda=lam_rho    
    
    if(x[1]==1):        
        
        return np.array([0,0,0])
        
    elif(x[0]<1e-10):#essentially zero and idential to main function
        return np.array([0,(x[1]-1) * ( FI_ab[1] * r  + FI_ab[2] * x[1]) ,  FI_ab[2]])
    else:
        
        return  np.array([ (x[1]-1)**2 * (FI_ab[0] * (r * a)**2 + 2 * r * a * b * FI_ab[1] + FI_ab[2] * b**2), (x[1]-1) * (FI_ab[0] * (r * a)**2 * x[0] + FI_ab[1] * r * a *( (1+x[0]) * foo +  b * x[0]) + FI_ab[2] * b *(1+x[0]) * foo),  FI_ab[0] * (r * a * x[0])**2 + 2 * r * a * x[0] *  (1+ x[0]) * foo *  FI_ab[1] + FI_ab[2] *((1+x[0]) * foo )**2]) * foo**2
    
 
 
def ComputeCov_NegBinom(Fisher_Info,Fisher_Info_r,sample_size,lam_rho,rho,lam,r,phi,q):
    

    #now we need to transform, we have original parameters (lambda,r,rho,q) 
    #1) transformed  into (lambda_rho,r,Lambda,q)  
    #2) transformed  into (lambda_rho,r,Phi,q)
    
    # hence  (AB) I (A B)^T 
    
    #3)  we compute the covariance matrix, which is the invers of the Fisher Information matrix.
    # the naive way is to compute the Invers of  (AB) I (A B)^T  but  since our Fisher Information can be singular (entries which are infinite )
    #we proceed by first Inverting and then multyipliying i.e. we compute  ((A B) I (AB)^T)^-1 =  ((AB)^T)^-1 ((AB) I)^-1 = (AB)^T)^-1  I^-1 (A B)^-1 =
    # =  (B^-1 A^-1)^T  I^-1 (B^-1 A^-1)
    
        
    AInvers=np.zeros((4,4),dtype=float)
    
    #A = [rho 0 1-rho 0
    #      0   1   0   0
    #     lam 0 -lam  0 
    #     0   0   0   1 ]
        
     #invers according to wolfram alpha   
    
    #A^-1 = [1 0 (1-rho)/lam 0
    #        0 1    0        0
    #        1 0 - rho/lam   0 
    #        0 0   0         1 ]
    
    AInvers[0,0]=1
    #AInvers[0,1]=0
    AInvers[0,2]=(1-rho)/lam
    #AInvers[0,3]=0
    #AInvers[1,0]=0
    AInvers[1,1]=1
    #AInvers[1,2]=0
    #AInvers[1,3]=0
    AInvers[2,0]= 1
    #AInvers[2,1]=0
    AInvers[2,2]=-rho/lam
    #AInvers[2,3]=0
    #AInvers[3,0]=0
    #AInvers[3,1]=0
    #AInvers[3,2]=0
    AInvers[3,3]=1
    
    
    #a and b just placeholders
     #B = [1 0 a 0
    #      0 1 a 0
    #      0 0 b 0 
    #      0 0 0 1]
    
    #invers according to wolfram alpha   
    #B^-1 = [1 0 -a/b 0
    #      0 1 -a/b 0
    #      0 0 1/b 0 
    #      0 0 0 1]
    
   

    BInvers=np.zeros((4,4),dtype=float)

    
    BInvers[0,0]=1
    BInvers[0,2]=  phi 
    BInvers[1,1]=1
    BInvers[1,2]=BInvers[0,2]
    BInvers[2,2]= r+lam_rho
    BInvers[3,3]=1
    

    Cov=np.zeros((4,4),dtype=float)
    #thats just placeholders

     #so 
    #Cov=FI^-1  = [1/a  0 0
    #              0   e/(ce-d^2) -d/(ce-d^2) 
    #              0   -d/(ce-d^2) c/(ce-d^2)]
        
    Cov[0,0]= lam_rho * (r+lam_rho) /(sample_size  *  r)
    Cov[1,1]=1/Fisher_Info_r 
    
    # all divergent if a=b=1 and hence here only if q=1
    # aa divergent for a =1, and hence here for q=1 (already covered) and lambda=lam_rho    
    
   # print('FI')
    #print(Fisher_Info)
    if(q==1):        
        
        Cov[2,2]=np.nan
        Cov[2,3]=np.nan
        Cov[3,2]=np.nan
        Cov[3,3]=np.nan
        
    elif(lam_rho==lam):#before set to be equal so we can actually compare float
        Cov[3,3]= 1/(sample_size * Fisher_Info[2]) 
    else:
        
    
        det= Fisher_Info[0] * Fisher_Info[2] - Fisher_Info[1]**2

        Cov[2,2]=  Fisher_Info[2]/(sample_size * det)
        Cov[2,3]= -Fisher_Info[1]/(sample_size * det)

        Cov[3,2]=  Cov[2,3]  
        Cov[3,3]= Fisher_Info[0]/(sample_size * det)
    

    T= np.matmul(BInvers,AInvers)    

    return np.matmul(np.transpose(T) ,np.matmul(Cov,T))



def OptimizeWithSeveralStartPostions_NegBinomial(num_neg,mean_num_det_neg,counts_num_det_pos,lam_rho,r):
    q_ini=np.arange(0.1,1,0.1)    
    Phi_ini=np.arange(0.1,10 * lam_rho,1)    

    optima=[]
    optima_val=[]

    #to be optimised by paralellisation or zipping 
    for Phi in Phi_ini:  
        foo=[optimize.minimize(Log_Liklihood_Coin_Toss_Stochastic_NegBinom,args=(r,num_neg,mean_num_det_neg,counts_num_det_pos), x0=[Phi, q],method='SLSQP',bounds=((0,None),(0,1)))
            for q in q_ini]

        LogL=[i.fun for i in foo]
        minpos = LogL.index(min(LogL))
        optima.append(foo[minpos].x)
        optima_val.append(min(LogL))
                          
    
    minpos = optima_val.index(min(optima_val))

    return optima[minpos]   
        

        

def Score_NegBinomial_r(n,r):
    
    # parametrised as mean value and r, estimate for mean value already plugged in
    
    
    foo=0
    for i in n:
        foo+=psi(r+i)

    N=len(n)
    return foo - N * psi(r) + N * (np.log(r) - np.log(r+ np.mean(n)))

def Estimate_Fisher_Information_part_r(lam_rho,r,num_samples):
    
       
    chip_1 = pd.DataFrame(np.random.negative_binomial(r,r/(lam_rho+r), num_samples),columns=['cells'])

    chip_1['estimate']=chip_1['cells'].apply(lambda n:  polygamma(1,r+n))
    
    return polygamma(1,r)-lam_rho/r * 1/(r+lam_rho) - chip_1['estimate'].mean()


def AnalyseNegBinomChip(chip):

    
    chip=chip.rename(columns={'n_cells': 'detected_cells'})   
    
    probDeath_detected = chip.groupby(['detected_cells']).dead.mean().reset_index()
    CountData_detected = chip.groupby(['detected_cells']).dead.count().reset_index()
    
    #single and zero detected as information
    foo = probDeath_detected.loc[probDeath_detected.detected_cells == 1]
    out=foo.loc[:]
    out = out.rename(columns={'dead': 'prob_neg_drop_one_det'})
        
    foo = probDeath_detected.loc[probDeath_detected.detected_cells == 0]
    out['prob_neg_drop_zero_det']=foo['dead'].values    

    
    #now all chip information
    out['prob_neg_drop']=chip.dead.mean()
    out['lambda_rho']=chip.detected_cells.mean()
    out['var']=chip.detected_cells.var()
    out['r']=np.power(out['lambda_rho'],2)/(out['var']-out['lambda_rho'])
        
        
    out['q_single']=out['prob_neg_drop_one_det']/np.power(out['prob_neg_drop_zero_det'],1/out['r']+1)
    foo=float(np.power(out['prob_neg_drop_zero_det']/out['prob_neg_drop'],1/out['r']))

    if float(1+out['lambda_rho']/out['r'] ) > foo:

        if foo > float(1+(out['lambda_rho']/out['r']) *(1-np.power(out['prob_neg_drop_zero_det'],1/out['r']))):

            out['q_chip']=(1+out['r']/out['lambda_rho'] * (1- foo))/np.power(out['prob_neg_drop_zero_det'],1/out['r'])    
        else:
            out['q_chip'] = 1

    else:
        out['q_chip'] = 0

    foo=float(np.power(1/out['prob_neg_drop'],1/out['r']))

    if float(1+out['lambda_rho']/out['r'] ) > foo:
        if foo > 1: 
            out['q_chip_uncor']=1+out['r']/out['lambda_rho'] * (1-np.power(1/out['prob_neg_drop'],1/out['r']))
        else:
            out['q_chip_uncor']=1

    else:
        out['q_chip_uncor']=0
        
    out['rho']= (1-out['q_chip']) * out['lambda_rho']/out['r'] * 1/(1/np.power(out['prob_neg_drop'],1/out['r'])-1)
    out['lambda']=out['lambda_rho']/out['rho']
    
    
    #--------------- now likelihood optimisation------------------------------------
    
        #---------------likelihood estimate for r --------------- 
    f = partial(Score_NegBinomial_r,chip.detected_cells)
    opt=optimize.root(f,out['r'].values[0], method='krylov') #we use moment estimate as intial value
    
    if(opt.success):
        out['r_like']=opt.x
    else:
        print('Root finding for r was not successful !')
        out['r_like']=np.nan
    
    
    mean_num_det_neg=chip.loc[chip['dead']==1,'detected_cells'].mean()
    counts_num_det_pos=np.bincount(chip.loc[chip['dead']==0,'detected_cells'])    
    out['Phi_like'],out['q_chip_like']= OptimizeWithSeveralStartPostions_NegBinomial(chip['dead'].sum(),mean_num_det_neg,counts_num_det_pos, out['lambda_rho'].values[0],out['r_like'].values[0])

    
    
    if(out['Phi_like'].values[0]< 1e-10): #essentially zero
        out['lambda_like']= out['lambda_rho']
        out['rho_like']=1

    else:

        out['lambda_like']= out['Phi_like'] * (out['r_like']+ out['lambda_rho']) + out['lambda_rho']
        out['rho_like']=out['lambda_rho']/out['lambda_like']
    
    # now Fisher Information
    num_samples=10000 # for Monte_Carlo estimate    
    Fisher_Information=   Fisher_Information_NegBinom([out['Phi_like'].values[0],out['q_chip_like'].values[0]],out['lambda_rho'].values[0],out['r_like'].values[0],num_samples)

    
    FisherInfo_r=  chip.shape[0] * Estimate_Fisher_Information_part_r(out['lambda_rho'],out['r_like'],num_samples) 

    Cov=ComputeCov_NegBinom(Fisher_Information,FisherInfo_r,chip.shape[0],out['lambda_rho'].values[0],out['rho_like'].values[0],out['lambda_like'].values[0]
                                                                        ,out['r_like'].values[0],out['Phi_like'].values[0],out['q_chip_like'].values[0])
                            
    
    out['var_lam_chip_like']=Cov[0,0]
    out['var_r_chip_like']=Cov[1,1]
    out['var_rho_chip_like']=Cov[2,2]
    out['var_q_chip_like']=Cov[3,3]
    
    out['cov_lam_r_chip_like']=Cov[0,1]
    out['cov_lam_rho_chip_like']=Cov[0,2]
    out['cov_lam_q_chip_like']=Cov[0,3]
    
    out['cov_r_rho_chip_like']=Cov[1,2]
    out['cov_r_q_chip_like']=Cov[1,3]
    
    out['cov_rho_q_chip_like']=Cov[2,3]
    
      
    del out["detected_cells"]
    print('############################# Exit Analysis-###########################')
    return out