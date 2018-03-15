import tensorflow as tf


get_variable2=tf.get_variable


def range_noisy(*args):
    for i in iterate_noisy(range(*args)):
        yield i


import time
import numpy as np
import sys


import numpy as np
import scipy as sp
import scipy.stats
import scipy.special

def iterate_noisy(lst):

    times=[]

    lst=list(lst)
    n=len(lst)

    assert n%1==0    
    n=int(n)
    assert n>0
    regularity=n//20
    if regularity<=0:
        regularity=1
    
    for i in range(n):
        yield lst[i]
        times.append(time.time())
        
        if i<5 or i%regularity==0:
            if len(times)>1:
                elapsed=times[-1]-times[0]
                meantime=np.mean(np.diff(times))

                completion_expected=(elapsed/i)*n
                time_remaining=(elapsed/i)*(n-i)

                # pnn("(%d⸗%.1f)"%(i,time_remaining),space=False)
                pnn("(%d⸗%.1f)"%(i,time_remaining/60.0),space=False)
                # pnn("(%d %.2f %.2f)"%(i,elapsed/60.0, (n-i)*meantime/60.0))
                # pnn("(%d:%.1f)"%(i,(n-i)*meantime/60.0))

                # print("%f elapsed, on iteration %d, completion expected %f, time remaining %f"%(
                    # elapsed,i,completion_expected,time_remaining))

        if i==n-1:
            pnn("(UTL⸗%1f)"%((times[-1]-times[0])/60.0))

def pnn(x,space=True): #{{{1
    ''' print no-newline '''
    sys.stdout.write(str(x) + (" " if space else ''))
    sys.stdout.flush()

'''
 _        _                               
| |_ __ _| |__   ___ ___  _ __ ___  _ __  
| __/ _` | '_ \ / __/ _ \| '_ ` _ \| '_ \ 
| || (_| | |_) | (_| (_) | | | | | | |_) |
 \__\__,_|_.__/ \___\___/|_| |_| |_| .__/ 
                                   |_|    
'''

class TabCompleteGraph:
    def __init__(self,gr,scope=''):
        self.gr=gr
        self.scope='/'+scope
        self.lenscope=len(self.scope)

    def __repr__(self):
        return ':%s:'%self.scope

    def __dir__(self):
        ops,subs=self.get_children(justname=True)

        return ops+subs

    def __getattr__(self,x):
        ops,subs=self.get_children()

        if x in ops:
            return ops[x]
        elif x in subs:
            return subs[x]
        else:
            raise AttributeError("huh?")

    def get_children(self,justname=False):
        names=self.gr._nodes_by_name.keys()

        names2=[]
        for nm in names:
            nm=('/'+nm)
            if nm[:self.lenscope]==self.scope:
                names2.append(nm[self.lenscope:])

        ops=[]
        subs=set()
        for nm in names2:
            if '/' in nm:
                subs.add(nm[:nm.index('/')])
            else:
                ops.append(nm)

        if justname:
            return ops,list(subs)
        else: 
            subs={sub:TabCompleteGraph(self.gr,scope=self.scope[1:]+sub+"/") for sub in subs}
            ops={nm:self.gr.get_operation_by_name(self.scope[1:]+nm) for nm in ops}

            return ops,subs

'''
 _           _       _                                
| |__   __ _| |_ ___| |__  _ __   ___  _ __ _ __ ___  
| '_ \ / _` | __/ __| '_ \| '_ \ / _ \| '__| '_ ` _ \ 
| |_) | (_| | || (__| | | | | | | (_) | |  | | | | | |
|_.__/ \__,_|\__\___|_| |_|_| |_|\___/|_|  |_| |_| |_|
                                                      
'''

def batch_normalize_linear(name,momentsource,target_mean,target_stdev):
    GR=tf.get_default_graph()
    momentsource_tensor=GR.get_tensor_by_name(momentsource+":0")
    
    with tf.variable_scope(name):
        wts=GR.get_tensor_by_name(momentsource+'_layer/kernel:0')
        bias=GR.get_tensor_by_name(momentsource+'_layer/bias:0')
    
        layshape1 = wts.shape[1]
        if layshape1.value is None:
            layshape1=tf.shape(wts)[1]

        mn,vr=tf.nn.moments(momentsource_tensor,axes=[0],name='moments')
        assig1 = tf.assign(wts,target_stdev * wts / tf.reshape(tf.sqrt(vr),[-1,layshape1]),name='rekernel')
        assig2 = tf.assign(bias,target_mean + bias-mn,name='rebias')

    return assig1,assig2

def set_limits_convex_combination(name,target,target_low,target_high):
    GR=tf.get_default_graph()
    
    with tf.variable_scope(name):
        bias1=GR.get_tensor_by_name(target+'_layer/bias1:0')
        bias2=GR.get_tensor_by_name(target+'_layer/bias2:0')
    
        assig1 = tf.assign(bias1,target_low,name='rebias1')
        
        with tf.control_dependencies([assig1]):
            assig2 = tf.assign(bias2,target_high,name='rebias2')

        assig = tf.identity(assig2,name='batchnorm')

    return assig


'''
 _                           
| | __ _ _   _  ___ _ __ ___ 
| |/ _` | | | |/ _ \ '__/ __|
| | (_| | |_| |  __/ |  \__ \
|_|\__,_|\__, |\___|_|  |___/
         |___/               

'''

def translate_initializer(init,dtype):
    if isinstance(init,float) or isinstance(init,int):
        return tf.constant_initializer(init)
    elif isinstance(init,tuple):
        return tf.random_uniform_initializer(minval=init[0],maxval=init[1],dtype=dtype)
    elif init=='glorot':
        return tf.contrib.layers.xavier_initializer(dtype=dtype)
    else:
        return init



def linear(name,inp,outsize,dtype,kernel_initializer='glorot',bias_initializer=0):
    with tf.variable_scope(name+"_layer"):
        inpsize=inp.shape[1]
        bias_initializer=translate_initializer(bias_initializer,dtype)
        kernel_initializer=translate_initializer(kernel_initializer,dtype)
        
        kernel = get_variable2('kernel',(inpsize,outsize),dtype,kernel_initializer)
        bias = get_variable2('bias',(outsize),dtype,bias_initializer)
        
        rez=tf.tensordot(inp,kernel,axes=[[1],[0]])+bias
    return tf.identity(rez,name=name)

def convex_combination(name,inp,outsize,dtype,bias1_initializer=(-1,1),bias2_initializer=(-1,1)):
    with tf.variable_scope(name+"_layer"):
        bias1_initializer=translate_initializer(bias1_initializer,dtype)
        bias2_initializer=translate_initializer(bias2_initializer,dtype)
        bias1 = get_variable2('bias1',(outsize),dtype,bias1_initializer)
        bias2 = get_variable2('bias2',(outsize),dtype,bias2_initializer)
        
        rez=inp*(bias1) + (1-inp)*(bias2)
    return tf.identity(rez,name=name)

def diag(name,inp,outsize,dtype,weight_initializer=(-1,1)):
    with tf.variable_scope(name+"_layer"):
        weight_initializer=translate_initializer(weight_initializer,dtype)
        weight = get_variable2('weight',outsize,dtype,weight_initializer)
        rez=inp*weight
    return tf.identity(rez,name=name)

def rbf(name,inp,outsize,minvar,dtype,center_initializer=(-1,1),scale_initializer=.5):
    with tf.variable_scope(name+"_layer"):
        inpsize=inp.shape[1]
        center_initializer=translate_initializer(center_initializer,dtype)
        scale_initializer=translate_initializer(scale_initializer,dtype)
        
        centers=get_variable2('centers',shape=(inpsize,outsize),dtype=dtype,initializer=center_initializer)
        stds=get_variable2('stds_almost',shape=(outsize,),dtype=dtype,initializer=scale_initializer)
        var=tf.identity(stds**2+minvar,name='vars')
        
        diff= tf.reshape(inp,(tf.shape(inp)[0],inpsize,1)) - tf.reshape(centers,(1,inpsize,outsize))
        diff = tf.identity(diff,name='diff')
        rez = tf.exp(-.5*tf.reduce_sum(diff**2/var,axis=1),name='rez')
    return tf.identity(rez,name)

'''
                 _           _     _ _ _ _         
 _ __  _ __ ___ | |__   __ _| |__ (_) (_) |_ _   _ 
| '_ \| '__/ _ \| '_ \ / _` | '_ \| | | | __| | | |
| |_) | | | (_) | |_) | (_| | |_) | | | | |_| |_| |
| .__/|_|  \___/|_.__/ \__,_|_.__/|_|_|_|\__|\__, |
|_|                                          |___/ 
'''

def ZINB_prob(x,alpha,beta,h_log_off,h_log_on,name,h_logit=None):
    '''
    We take 
    
    - x [batch x Ng]
    - alpha [batch x Ng]
    - beta [batch x Ng]
    - h_log_off [batch x Ng]
    - h_log_on [batch x Ng]
    
    If provided, we assume h_logit = h_log_on - h_log_off.
    
    We calculate a mixture between 

    - delta_0 (with probability exp(h_log_off))
    - NBinom(alpha,beta) (with probability exp(h_log_on))
    
    Here by NBinom we mean take
    
       PP = 1 / (1+beta)
       binomcoef(alpha + x - 1,x) * (1-PP)^alpha p^x
    
    '''

    with tf.variable_scope(name+"_computation"):
    
        ################## get some stuff we need
        logPP = -tf.log1p(beta) # log(PP)
        log1mPP = tf.log(beta) + logPP # log(1-PP)
        
        ################# nbinom
        # gamma term
        NB = tf.lgamma(x+alpha) - tf.lgamma(x+1) - tf.lgamma(alpha)
        
        # ps
        NB = NB + alpha*log1mPP + x*logPP
        
        ############ put it with the mixture
        # let p the probability of going ON
        # we need log((1-p)*(x==0) + p*exp(NB))
        # so we divide this into two cases
        
        # case I: x==0
        #   log((1-p)+ p*exp(NB))
        # = log(1 + p*exp(NB)/(1-p)) + log(1-p)
        # = softplus(NB+h_logit) + log(1-p)
        caseI = tf.identity(tf.nn.softplus(NB+h_logit) + h_log_off,name='caseI')
        
        # case II
        #  log(p*exp(NB)) = log(p) + NB
        caseII = tf.identity(NB + h_log_on,name='caseII')
        
        # switch between them accordingly
        SWITCH = tf.cast(tf.equal(x,0),x.dtype,name='SWITCH')
        rez = caseI*SWITCH+caseII*(1-SWITCH)
        
    ################# return 
    return tf.identity(rez,name=name)

def kl_categorical(a,lp_a,lp_b,name,formula=None):
    '''
    reduce_sum(a * (lp_a-lp_b))
    '''
    
    with tf.variable_scope(name+"_computation"):

        if formula is not None:
            a,lp_a,lp_b = einop(formula,'einop',a,lp_a,lp_b)
        rez = a*(lp_a-lp_b)
            
    return tf.identity(rez,name)

def kl_gaussian(mu1,sigma1sq,mu2,sigma2sq,name,formula=None):
    '''
    .5*reduce_sum((mu1 - mu2)^2 / (sigma2sq) + sigma1sq/sigma2sq - 1 - log(sigma1sq/sigma2sq))
    '''
    
    with tf.variable_scope(name+"_computation"):
   
        # for x in mu1,sigma1sq,mu2,sigma2sq:
        #     print(x)

        if formula is not None:
            mu1,sigma1sq,mu2,sigma2sq= einop(formula,'einop',mu1,sigma1sq,mu2,sigma2sq)


        # for x in mu1,sigma1sq,mu2,sigma2sq:
        #     print(x)
        
        ooss = 1.0/sigma2sq
        
        rez = .5*(ooss*(mu1-mu2)**2 + ooss*sigma1sq - 1 - tf.log(sigma1sq*ooss))
    return tf.identity(rez,name)

'''
      _                   
  ___(_)_ __   ___  _ __  
 / _ \ | '_ \ / _ \| '_ \ 
|  __/ | | | | (_) | |_) |
 \___|_|_| |_|\___/| .__/ 
                   |_|    
'''

def einop(eqn,name,*inps):
    with tf.variable_scope(name):
        order,shapes,grs=einop_pure(eqn)
        assert len(shapes)==len(inps),"%s VS %s"%(str(shapes),str(inps))

        # an elementary check
        for i in range(len(inps)):
            if len(grs[i])!=len(inps[i].shape):
                raise Exception("formula '%s' for input %s (%s) suggests a %d-tensor, but instead found a %d-tensor"%(
                    grs[i],inps[i].name,inps[i].shape,len(grs[i]),len(inps[i].shape)))

        nobs = len(inps)
        ndims = len(order)
        
        local_to_globals=[] # for each input, maps a localaxis -> globalaxis
        dimassigs=[] # for each input, maps an axislabel -> dimensionsize

        outs=[]

        # for each input
        for i in range(nobs):
            dimassig={}
            dimassigs.append(dimassig)
            local_to_global={}
            local_to_globals.append(local_to_global)


            myshape=[]
            mytensorshape = tf.shape(inps[i],name='inp%02d_shape'%i)
            mytensorshape2 = inps[i].shape

            for j in range(ndims):
                
                # The jth dimension of the final object
                # will be filled by the corresponding dimension of inp
                corresponding_dimension = shapes[i][j]
                
                if corresponding_dimension==-1: 
                    myshape.append(1) # expand_dims
                    dimassig[order[j]]='one'
                else:
                    # abstractly, myshape.append(inps[i].shape[corresponding_dimension])
                    local_to_global[corresponding_dimension]=j
                    
                    knownvalue=mytensorshape2[corresponding_dimension].value
                    if knownvalue is None:
                        dimassig[order[j]]='dar'
                        myshape.append(mytensorshape[corresponding_dimension])
                    else:
                        dimassig[order[j]]=knownvalue
                        myshape.append(knownvalue)
            outs.append(tf.reshape(inps[i],myshape,name='out%02d'%i))

    # some elementary checks:
    # for i in range(len(dimassigs)):

    # print(inps[i].name,dimassigs[i],local_to_globals[i])

    # # attempt to determine the size of each dimension
    # sizes=[]
    # for i in range(len(order)):
    #     SZ=None
    #     ORIGIN=None
    #     for j in range(len(outs)):
    #         if outs[j].shape[i].value is not None:
    #             SZ2 = outs[j].shape[i].value
    #             if SZ is None:
    #                 if (SZ2 is not None) and (SZ2>1):
    #                     SZ=SZ2
    #                     ORIGIN=j
    #             else:
    #                 if SZ2>1 and SZ!=SZ2:
    #                     raise Exception("Input %s(%s) and %s(%s) disagree about sizes: %s vs %s"%(
    #                         inps[ORIGIN].name,inps[ORIGIN].shape,
    #                         inps[j].name,inps[j].shape,
    #                         outs[ORIGIN].shape,outs[j].shape))


    return outs
            

def einop_pure(eqn):
    # get the axes
    eqn=eqn.split('->')
    grs = eqn[0].split(',')
    gr_final = eqn[1]

    # make sure that everyone is a set
    for gr in grs+[gr_final]:
        assert len(gr)==len(set(list(gr)))

    # make sure that gr_final includes everyone
    for i,gr in enumerate(grs):
        if not set(gr).issubset(set(gr_final)):
            raise Exception("Formula %s for input %d includes a character not found in output shape %s"%(
                gr,i,gr_final))

    # for each axis of each input, find out where that axis lies in the final output
    input_shape=[[gr.find(x) for x in gr_final] for gr in grs]

    # make sure no transpositions are required
    for i in range(len(grs)):
        cur_gr_index=0
        cur_final_index = input_shape[i][0]

        for m,j in enumerate(input_shape[i]):
            if j==-1:
                pass
            else:
                if j<cur_gr_index:
                    raise Exception("To apply input %s into output %s, we would need to transpose %s and %s"%(
                        grs[i],gr_final,grs[i][cur_gr_index],grs[i][m]))
                
                cur_gr_index=m
                cur_final_index=j
    
    return gr_final,input_shape,grs


'''
       _       _   
 _ __ | | ___ | |_ 
| '_ \| |/ _ \| __|
| |_) | | (_) | |_ 
| .__/|_|\___/ \__|
|_|                

'''

import matplotlib.pylab as plt
def figsize(n,m=None):
    if m is None:
        m=n
    plt.gcf().set_size_inches(n,m)
    
class AnimAcross:
    def __init__(self,total,columns=None,rows=None,sz=None,height=1.0):
        self.columns=columns
        self.rows=rows
        self.sz=sz
        self.total=total
        self.height=height

        if (self.columns is None) and (self.rows is None):
            self.columns=self.total
            self.rows=1
        elif self.columns is None:
            # we have `total` guys to split up 
            # amongst `rows` slots
            if total%rows==0: # a perfect fit!
                self.columns=total//rows
            else:
                self.columns = int(total/rows)+1
        elif self.rows is None:
            # we have `total` guys to split up 
            # amongst `column` slots
            if total%columns==0: # a perfect fit!
                self.rows=total//columns
            else:
                self.rows = int(total/columns)+1
        else:
            assert total==rows*columns

    def __enter__(self):
        if self.sz is not None:
            figsize(self.sz*self.columns,self.sz*self.rows*self.height)
        self.i=1
        return self

    def g(self):
        plt.subplot(self.rows,self.columns,self.i)
        self.i=self.i+1

    def __exit__(self,exc_type,exc_val,exc_tb):
        if exc_type is not None:
            print(exc_type,exc_val,exc_tb)
            
'''
  __                  _             _    
 / _|_   _ _ __   ___| | ___   ___ | | __
| |_| | | | '_ \ / __| |/ _ \ / _ \| |/ /
|  _| |_| | | | | (__| | (_) | (_) |   < 
|_|  \__,_|_| |_|\___|_|\___/ \___/|_|\_\
                                         

'''

def funclook(proj2d, vals,bins,defsamp=None):
    '''
    Let proj2d be a (n x 2) matrix indicating positions
     and vals be an n-vector indicating values.
    
    For each bin defined by bins,
    we take a random sample of vals among vals such that proj2d is in that bin, or nan if there is nothing there.
    along the way, we compute means, counts, and standard deviations
    '''


    sampx=proj2d[:,0]
    sampy=proj2d[:,1]

    xs,ys = bins
    nbins=len(xs),len(ys)


    # for each item in 1...n, find out which bin it should be assigned to
    assigsX=np.searchsorted(xs,proj2d[:,0])-1
    assigsY=np.searchsorted(ys,proj2d[:,1])-1

    good = (assigsX>-1) & (assigsX<nbins[0]-1) & (assigsY>-1) & (assigsY<nbins[1]-1)

    assigsX2=assigsX[good]
    assigsY2=assigsY[good]
    vals2=vals[good]

    means_accum=np.zeros((nbins[0]-1,nbins[1]-1))
    stds_accum=np.zeros((nbins[0]-1,nbins[1]-1))
    samples=np.full((nbins[0]-1,nbins[1]-1),np.nan)
    count=np.zeros((nbins[0]-1,nbins[1]-1))

    np.add.at(means_accum,(assigsX2,assigsY2),vals2)
    np.add.at(stds_accum,(assigsX2,assigsY2),vals2**2)
    np.add.at(count,(assigsX2,assigsY2),1)

    good2 = count>0

    means=np.full((nbins[0]-1,nbins[1]-1),np.nan)
    means[good2] = means_accum[good2] / count[good2]
    stds=np.full((nbins[0]-1,nbins[1]-1),np.nan)
    variances = (stds_accum[good2] / count[good2]) - means[good2]**2
    variances[variances<0]=0
    stds[good2] = np.sqrt(variances)

    means[~good2]=np.mean(means[good2])
    stds[~good2]=np.mean(stds[good2])

    samples[assigsX2,assigsY2]=vals2
    
    if defsamp is None:
        samples[~good2]=np.mean(samples[good2])
    else:
        samples[~good2]=defsamp 

    return means,stds,count,samples

'''
                               _             
 _ __  _ ____   _____ _ __ ___(_) ___  _ __  
| '_ \| '_ \ \ / / _ \ '__/ __| |/ _ \| '_ \ 
| | | | |_) \ V /  __/ |  \__ \ | (_) | | | |
|_| |_| .__/ \_/ \___|_|  |___/_|\___/|_| |_|
      |_|              
'''

# as a code test
# we reimplement the loss function in numpy
# should make sure we get the same answers in tf and np

def safe_softplus(x, limit=30):
    rez=np.zeros(x.shape,x.dtype)
    rez[x>limit] = x[x>limit]
    rez[x<=limit] = np.log(1+np.exp(x[x<=limit]))
    return rez


def np_ZINB_loss(data,alpha,beta,h):
    '''
    returns log probability of data under a mixture of 
    negative binomial with a (1-h) probability of delta0
    '''

    scipy_pp = beta / (1+beta)
    nbprob = sp.stats.nbinom.logpmf(data,n=alpha,p=scipy_pp)

    # abstractly, this: 
    # return np.log(np.exp(nbprob)*h + (1-h)*(data==0))

    # in practice, this:
    # when data is 0   :::: np.log(np.exp(nbprob)*h/(1-h) + 1) + np.log(1-h)
    # when data is !0  :::: nbprob + log(h)

    case_I = safe_softplus(nbprob+np.log(h/(1-h))) + np.log(1-h)
    case_II = nbprob + np.log(h)

    rez=(data==0) * case_I + (data!=0) * case_II
    return rez


def np_kl_categorical(pi1,pi2):
    return pi1 * (np.log(pi1) - np.log(pi2))

def np_kl_gaussian(mu1,var1,mu2,var2):
    return ((mu1-mu2)**2 / (2*var2)) + .5*((var1/var2) -1 - np.log(var1/var2))

def np_loss(data,muT,varT,pi,mu1tilde,var1tilde,mu2tilde,var2tilde,pitilde,
    alphaT,betaT,hT):
    '''
    For testing against the tensorflow implementation....

    data: vector of genes (Ng,)
    muT: vector of means (Nclust,Nk)
    varT: vector of vars (Nclust,)
    pi: mixing (Nclust,)
    mu1tilde: (Nk,)
    var1tilde: ()
    mu2tilde: (Nclust,Nk)
    var2tilde: (Nclust,)
    pitilde: (Nclust,)
    w1: (Nk,)
    w2: (Nk,)
    alphaT: (Nclust,Ng,)
    betaT: (Nclust, Ng,)
    hT: (Nclust,Ng,)
    '''

    Nclust,Nk = muT.shape
    Ng,=data.shape

    print("Nclust",Nclust,"Nk",Nk,'Ng',Ng)


    pi_kl_loss = np.sum(np_kl_categorical(pitilde,pi))

    nbs = np.zeros((Nclust,Ng))
    kl1s = np.zeros((Nclust,Nk))
    kl2s = np.zeros((Nclust,Nk))
    for t in range(Nclust):
        nbs[t]=(np_ZINB_loss(data,alphaT[t],betaT[t],hT[t]))
        kl1s[t]=(np_kl_gaussian(mu1tilde,var1tilde,muT[t],varT[t]))
        kl2s[t]=(np_kl_gaussian(mu2tilde[t],var2tilde[t],0,1))

    total = pi_kl_loss + np.sum(pitilde*(-np.sum(nbs,axis=1)+np.sum(kl1s+kl2s,axis=1)))


    return total/np.prod(data.shape),pi_kl_loss,nbs,kl1s,kl2s




