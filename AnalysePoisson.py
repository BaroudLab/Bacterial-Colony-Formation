import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

def ComputeCov_Poiss(Fisher_Info,sample_size,lam_rho,rho,lam,q):
    

    #now we need to transform, we have original parameters (lambda,r,rho,q) 
    #1) transformed  into (lambda_rho,r,Lambda,q)  
    
    # hence  (A) I (A)^T 
    
    #3)  we compute the covariance matrix, which is the invers of the Fisher Information matrix.
    # the naive way is to compute the Invers of  (A) I (A )^T  but  since our Fisher Information can be singular (entries which are infinite )
    #we proceed by first Inverting and then multyipliying i.e. we compute   (A I A^T)^-1 =  (A^T)^-1 (A I)^-1 = (A^T)^-1 I^-1 A^-1 =
    # (A^-1)^T I^-1 A^-1   
    
    AInvers=np.zeros((3,3),dtype=float)
    
    #A = [rho  1-rho 0
    #     lam  -lam  0 
    #       0   0   1 ]
            
    #A^-1 = [1 (1-rho)/lam 0
    #        1 -rho/lam   0 
    #        0    0         1 ]
    
    AInvers[0,0]=1
    AInvers[0,1]=(1-rho)/lam

    AInvers[1,0]=1
    AInvers[1,1]=-rho/lam
    AInvers[2,2]=1
    
   

    Cov=np.zeros((3,3),dtype=float)
    #thats just placeholders

    #FI = [a 0 0 
    #      0  c d 
    #      0  d e]

     #so 
    #Cov=FI^-1  = [1/a  0 0
    #              0   e/(ce-d^2) -d/(ce-d^2) 
    #              0   -d/(ce-d^2) c/(ce-d^2)]
        
    Cov[0,0]= lam_rho/sample_size
        
    # all divergent if a=b=1 and hence here only if q=1
    # aa divergent for a =1, and hence here for q=1 (already covered) and lambda=lam_rho    
    
   # print('FI')
    #print(Fisher_Info)
    if(q==1):        
        
        Cov[1,1]=np.nan
        Cov[1,2]=np.nan
        Cov[2,1]=np.nan
        Cov[2,2]=np.nan
        
    elif(lam_rho==lam):#before set to be equal so we can actually compare float
        Cov[2,2]= 1/(sample_size * Fisher_Info[2]) 
    else:
        
    
        det= Fisher_Info[0] * Fisher_Info[2] - Fisher_Info[1]**2

        Cov[1,1]=  Fisher_Info[2]/(sample_size * det)
        Cov[1,2]= -Fisher_Info[1]/(sample_size * det)

        Cov[2,1]=  Cov[1,2]  
        Cov[2,2]= Fisher_Info[0]/(sample_size * det)

    return np.matmul(np.transpose(AInvers) ,np.matmul(Cov,AInvers))
    


def Estimate_Fisher_Information_Poiss(x,lam_rho,num_samples):
      
        
    chip_1 = pd.DataFrame(np.random.poisson(lam_rho, num_samples),columns=['cells'])
    chip_2 = pd.DataFrame(np.random.poisson(lam_rho, num_samples),columns=['cells'])
    chip_3 = pd.DataFrame(np.random.poisson(lam_rho, num_samples),columns=['cells'])     
        
    chip_1['aa_estimate']=chip_1['cells'].apply(lambda n: np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )
    chip_2['ab_estimate']=chip_2['cells'].apply(lambda n: n*  np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )
    chip_3['bb_estimate']=chip_3['cells'].apply(lambda n: n**2 * np.power(x[1],n)/(1-x[0] * np.power(x[1],n)) )

        
    return [ chip_1['aa_estimate'].mean()/x[0], chip_2['ab_estimate'].mean()/x[1],(x[0]/x[1]**2) * chip_3['bb_estimate'].mean()]  



def Log_Liklihood_Coin_Toss_Stochastic_Poisson(x,num_neg,mean_num_det_neg,counts_num_det_pos):

    #input is Lambda and q
    a= np.exp(-(1-x[1])*x[0])
  
    return Log_Liklihood_Coin_Toss_Stochastic([a,x[1]],num_neg,mean_num_det_neg,counts_num_det_pos)
    
    
def  Score_Coin_Toss_Stochastic_Poisson(x,num_neg,mean_num_det_neg,counts_num_det_pos):
    
    #input is Lambda and q
    a= np.exp(-(1-x[1])*x[0])
    score_ab=Score_Coin_Toss_Stochastic([a,x[1]],num_neg,mean_num_det_neg,counts_num_det_pos) 
    return a  * np.array([(1-x[0]) * score_ab[0]  , x[0] *  score_ab[0] + score_ab[1]])

    
def Fisher_Information_Poiss(x,lam_rho,num_samples):
  
    #input is Lambda and q
    a= np.exp(-(1-x[1])*x[0])


    FI_ab = Estimate_Fisher_Information_Poiss([a,x[1]],lam_rho,num_samples)
    
    
    # all FI divergent if a=b=1 and hence here only if q=1
    # aa divergent for a =1, and hence here for q=1 (already covered) and lambda=lam_rho    
    if(x[1]==1):        
        return np.array([0,0,0])
    elif(x[0]<1e-10):#essentially zero and idential to main function
        return  np.array([0, (x[1]-1) * FI_ab[1], FI_ab[2]])
    else:
        
        return  np.array([ (a * (1-x[1]))**2 * FI_ab[0] , a * (x[1]-1) * (a *  x[0] * FI_ab[0] + FI_ab[1]),(a * x[0])**2 * FI_ab[0] + 2 * a * x[0] * FI_ab[1] + FI_ab[2]])
    

    
def OptimizeWithSeveralStartPostionsPoiss(num_neg,mean_num_det_neg,counts_num_det_pos, lam_rho):
    
    Lam_ini=np.arange(0.1,10 * lam_rho,1)    
    q_ini=np.arange(0.1,1,0.1)    
    
    optima=[]
    optima_val=[]

    #to be optimised by paralellisation or zipping 
    for Lam in Lam_ini:  
        foo=[optimize.minimize(Log_Liklihood_Coin_Toss_Stochastic_Poisson,args=(num_neg,mean_num_det_neg,counts_num_det_pos), x0=[Lam, q],method='SLSQP',bounds=((0,None),(0,1)))
            for q in q_ini]

        LogL=[i.fun for i in foo]
        minpos = LogL.index(min(LogL))
        optima.append(foo[minpos].x)
        optima_val.append(min(LogL))
                          
    
    minpos = optima_val.index(min(optima_val))

    return optima[minpos]   
        

def AnalysePoissonianChip(chip):
    chip=chip.rename(columns={'n_cells': 'detected_cells'})   

    probDeath_detected = chip.groupby(['detected_cells']).dead.mean().reset_index()
    CountData_detected = chip.groupby(['detected_cells']).dead.count().reset_index()
    
    #single and zero detected as information
    
    foo = probDeath_detected.loc[probDeath_detected.detected_cells == 1]
    out=foo.loc[:]
    out = out.rename(columns={'dead': 'prob_neg_drop_one_det'})
        
    foo = probDeath_detected.loc[probDeath_detected.detected_cells == 0]
    if(foo['dead'].values ):
        out['prob_neg_drop_zero_det']=foo['dead'].values    

    else:
        out['prob_neg_drop_zero_det']=np.Nan  

    out['q_single']=out['prob_neg_drop_one_det']/out['prob_neg_drop_zero_det']
    
    #now all chip information
    out['prob_neg_drop']=chip.dead.mean()
    out['lambda_rho']=chip.detected_cells.mean()
    out['rho']= np.log(out['prob_neg_drop']/out['prob_neg_drop_zero_det'])/np.log(out['prob_neg_drop'])
    out['lambda']=out['lambda_rho']/out['rho']
    out['var']=chip.detected_cells.var()

    
    
    if(float(out['prob_neg_drop']) < float(np.exp(-out['lambda_rho']))):
        out['q_chip_uncor']=0
    else:
        out['q_chip_uncor']=1+np.log(out['prob_neg_drop'])/out['lambda_rho']

    
    
    if(float(out['prob_neg_drop']/out['prob_neg_drop_zero_det']) < float(np.exp(-out['lambda_rho']))):
        out['q_chip']=0
    else:
        out['q_chip']=1-(np.log(out['prob_neg_drop_zero_det'])-np.log(out['prob_neg_drop']))/out['lambda_rho']

    
    mean_num_det_neg=chip.loc[chip['dead']==1,'detected_cells'].mean()
    counts_num_det_pos=np.bincount(chip.loc[chip['dead']==0,'detected_cells'])    

    out['Lam_like'],out['q_chip_like']= OptimizeWithSeveralStartPostionsPoiss(chip['dead'].sum(),mean_num_det_neg,counts_num_det_pos, out['lambda_rho'].values[0])
    
    if(out['Lam_like'].values[0]< 1e-10): #essentially zero
        out['lambda_like']= out['lambda_rho']
        out['rho_like']=1

    else:

        out['lambda_like']= out['Lam_like'] + out['lambda_rho']
        out['rho_like']=out['lambda_rho']/out['lambda_like']

    
    #now Fisher Information
    num_samples=10000 # for Monte_Carlo estimate
    Fisher_Information=Fisher_Information_Poiss([out['Lam_like'].values[0],out['q_chip_like'].values[0]],out['lambda_rho'].values[0],num_samples)
    

    # Fisher information for Poisson with parameter lambda is 1/lambda,
    #so we estimate it by pluggin in the estimate for lambda, which is identical to the observed fisher information
    
    Cov=ComputeCov_Poiss(Fisher_Information,chip.shape[0],out['lambda_rho'].values[0],out['rho_like'].values[0],out['lambda_like'].values[0],out['q_chip_like'].values[0])
   
    out['var_lam_chip_like']=Cov[0,0]
    out['var_rho_chip_like']=Cov[1,1]
    out['var_q_chip_like']=Cov[2,2]
    
    out['cov_lam_rho_chip_like']=Cov[0,1]
    out['cov_lam_q_chip_like']=Cov[0,2]
    
    out['cov_rho_q_chip_like']=Cov[1,2]
    
    del out["detected_cells"]

    return out    

