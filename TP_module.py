
# coding: utf-8

# In[1]:

import numpy as np
import pymc

print("TP_module is imported")

def fittingdata(filename,cutoff,binsize,upperlimit):
    '''
    General Overview: This makes the normalized-histogram of the NPDFs for a specific binsize
    
    Inputs: filename of the model
          : Cutoff by eys estimate to determine the power-law (optional any number will do)
          : Prefered binsize
          : Upper limit till which the date should be plotted (optional only needed for specific cases) 
          
    Output: Xdata (bincenter)
          : ydata normalized histogram size
          : Xdata for the lognormal part (optional)
          : Ydata for the lognormal part (optional)
    
    '''
    
    
        
    fp = open(filename, 'r')
    mass = []
    for line in fp:
        t = line.strip().split()
        for value in t:
            mass.append(float(value))
    
    fp.close()
    
    columndensity_array = np.asarray(mass)
#     numberdensity_array = columndensity_array * 1.5 *10.0**(21)
    
  
    bincount,bin_edge = np.histogram(np.log10(columndensity_array),binsize,density=True) ## i.e. divided by binwidth*N_total
    binwidth = (bin_edge[2]-bin_edge[1])
    N_total = len(columndensity_array)
    
    ## the total normalization is binwidth*N_total*np.log(10)
    ## the np.log(10) is multiplied in the fuction directly
    
    norm_const = binwidth*N_total ## this is the normalization constant
    
    #bincount,bin_edge = np.histogram(np.log10(numberdensity_array),binsize,density=False) 
    bincenter = (bin_edge[1:]+bin_edge[:-1])/2.0
    xdata=bincenter[:]
    
    ## this is added to eliminate the zero counts in the ydata    
       
    ydata_nonzero=[]
    for index, item in enumerate(bincount*norm_const):
#         if (item !=0 and item !=1):
        if (item >=1):
            ydata_nonzero.append(index)
#     print(ydata_nonzero)

    xdata_new=xdata[ydata_nonzero]  ## modified Xdata
    bincount_new=bincount[ydata_nonzero]  ## modified ydata
    
    ## this is added to eliminate the data beyond a range in the value of the column density
    if upperlimit is not None: 
#         print("with the upper limit")
        xdata_lessthan=[]
        for index, item in enumerate(xdata_new):
            if (item <=upperlimit):
                xdata_lessthan.append(index)    
    # print(ydata_nonnan)

        xdata_final=xdata_new[xdata_lessthan]
        ydata_final=bincount_new[xdata_lessthan] 

        #ydata= np.log10(bincount_new[:]/np.log(10))
    #     ydata= np.log10(ydata_final[:]/np.log(10))
        ydata= np.log10(ydata_final[:]) ### normalized but not divided by log10
        ydata_not_normalized = np.log10(ydata_final[:]*norm_const) 
        
        return xdata_final,ydata,ydata_not_normalized,norm_const
    
    else:
        print("c")
        ydata= np.log10(bincount_new) ### normalized but not divided by log10
        ydata_not_normalized = np.log10(bincount_new*norm_const)
        xdata_final = xdata_new

        if cutoff is not None: 
            x_min = cutoff
            powerlawzone = [i for i in xdata_final if i >= x_min]
            powerlawzonearray =(np.asarray(powerlawzone)) 
        #     ##############################Log Normal region #################
            # to get thex and y data of the lognormal zone

            lognormal_index =[]
            for index, value in enumerate(xdata_final):
                if (value <=x_min):
                    lognormal_index.append(index)
            lognormal_xdata = xdata_final[lognormal_index]
            lognormal_ydata = ydata_not_normalized[lognormal_index]
            
            return lognormal_xdata,lognormal_ydata,norm_const




        return xdata_final,ydata,ydata_not_normalized,norm_const


# In[4]:
constant_powlaw = []

def function1(x, x0, sigma, alpha, TP_min):
    
    '''
    General Overview: This is the log10 of the Piecewise lognormal and the power-law fucntion.
                      For data less then TP the function is lognormal and for
                      greater than TP its powerlaw
    
    Inputs: xdata
          
          
    Output: mean 
          : sigma
          : alpha
          : TP or transtion point
    
    '''
    xG = np.array([xx for xx in x if xx <= TP_min])
    def F1(xG):        
         p1 = np.log10(np.log(10)*1/(np.sqrt(2*np.pi)*sigma))
         p2 = np.log(10**xG)
         p3 = p1 - (p2 - x0)**2 / (2*np.log(10)*sigma**2)
         return p3
    xL = np.array([xx for xx in x if xx > TP_min])
    global A2
    A2 = F1(TP_min) - alpha*TP_min        
    constant_powlaw.append(A2) # To store the value of C
    def F2(xL):
        return alpha*xL + A2
    return np.concatenate((F1(xG), F2(xL)))

def function_lognormal(x, x0, sigma):
    
    '''
    General Overview: This is the log10 of the  lognormal.
                      This function will be called when the Tp value for the piece wise will be g
                      greater then then max- xdata value
    
    Inputs: xdata
          
          
    Output: mean 
          : sigma
          : alpha
          : TP or transtion point
    
    '''
    xG = x
    def F1(xG):        
         p1 = np.log10(np.log(10)*1/(np.sqrt(2*np.pi)*sigma))
         p2 = np.log(10**xG)
         p3 = p1 - (p2 - x0)**2 / (2*np.log(10)*sigma**2)
         return p3
    return F1(xG)
    
 
        
    
def model(x, f, p0):
#     A1 = pymc.Uniform("A1", 2.,6., value = p0[0])
    x0 = pymc.Uniform("x0", p0[0]-.4, p0[0]+.4, value = p0[0])
    sigma = pymc.Uniform("sigma", -0.01, 1., value = p0[1])
    alpha = pymc.Uniform("alpha", -7., 0., value = p0[2])
    TP_min = pymc.Uniform("TP_min", p0[3]-.4, p0[3]+.8, value = p0[3])

    @pymc.deterministic(plot = False)
    def function(x=x, x0=x0, sigma=sigma, alpha=alpha,TP_min=TP_min):
        xG = np.array([xx for xx in x if xx <= TP_min])
        def F1(xG):
            p1 = np.log10(np.log(10)*1/(np.sqrt(2*np.pi)*sigma))
            p2 = np.log(10**xG)
            p3 = p1 - (p2 - x0)**2 / (2*np.log(10)*sigma**2)
            return p3
        xL = np.array([xx for xx in x if xx > TP_min])
        A2 = F1(TP_min) - alpha*TP_min        
        def F2(xL):
            return alpha*xL + A2
        return np.concatenate((F1(xG), F2(xL)))
    s = pymc.HalfNormal('s',tau=1)
#         return F1(xG)
#     y = pymc.Normal("y", mu=function,tau = 1./f_err**2, value = f, observed = True)
    y = pymc.Normal("y", mu=function,tau = 1/s**2, value = f, observed = True)
    return locals()

### this is for plotting the unnormalized function


def function_plot(x, x0, sigma, alpha, TP_min,norm_const):
    
    '''
    General Overview: This is the log10 of the Piecewise lognormal and the power-law function not normalized
                      For data less then TP the function is lognormal and for
                      greater than TP its powerlaw
    
    Inputs: xdata
          
          
    Output: mean 
          : sigma
          : alpha
          : TP or transtion point
    
    '''
    xG = np.array([xx for xx in x if xx <= TP_min])
    def F1(xG):        
         p1 = np.log10(norm_const*np.log(10)*1/(np.sqrt(2*np.pi)*sigma))
         p2 = np.log(10**xG)
         p3 = p1 - (p2 - x0)**2 / (2*np.log(10)*sigma**2)
         return p3
    xL = np.array([xx for xx in x if xx > TP_min])
    global A2
    A2 = F1(TP_min) - alpha*TP_min        
    constant_powlaw.append(A2) # To store the value of C
    def F2(xL):
        return alpha*xL + A2
    return np.concatenate((F1(xG), F2(xL)))


def lognormal_plot(x, x0, sigma,norm_const):
    
    '''
    General Overview: This is the log10 of the  lognormal.
                      This function will be called when the Tp value for the piece wise will be g
                      greater then then max- xdata value
    
    Inputs: xdata
          
          
    Output: mean 
          : sigma
          : alpha
          : TP or transtion point
    
    '''
    xG = x
    def F1(xG):        
         p1 = np.log10(norm_const*np.log(10)*1/(np.sqrt(2*np.pi)*sigma))
         p2 = np.log(10**xG)
         p3 = p1 - (p2 - x0)**2 / (2*np.log(10)*sigma**2)
         return p3
    return F1(xG)


# def fitting_datafor_chisquare(xfit,yfit):
def fitting_datafor_chisquare(xfit,x0,sigma,norm_const):
    
    yfit = lognormal_plot(xfit,x0,sigma,norm_const)
    ydata_nonzero=[]
    for index, item in enumerate(yfit):
        if (item >=0):
            ydata_nonzero.append(index)
    yfit_new=yfit[ydata_nonzero]
    xfit_new =xfit[ydata_nonzero]
    xmax = np.max(xfit_new)
    xmin = np.min(xfit_new)
    return xmax,xmin

def fitting_datafor_chisquare_lgpw(xfit,x0, sigma, alpha, TP_min,norm_const):
    
    yfit = function_plot(xfit, x0, sigma, alpha, TP_min,norm_const)
    ydata_nonzero=[]
    for index, item in enumerate(yfit):
        if (item >=0):
            ydata_nonzero.append(index)
    yfit_new=yfit[ydata_nonzero]
    xfit_new =xfit[ydata_nonzero]
    xmax = np.max(xfit_new)
    xmin = np.min(xfit_new)
    return xmax,xmin


def readparamterevolution(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    mean_list = []  # units cm 
    sigma_list= []      # units ms-1
    alpha_list = []
    Transition_list = []
    meanerror_list = []  # units cm 
    sigmaerror_list= []      # units ms-1
    alphaerror_list = []
    Transitionerror_list = []
    Time_list = []
    

    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        mean_list.append(float(t[0]))
        meanerror_list.append(float(t[1]))
        sigma_list.append(float(t[2]))
        sigmaerror_list.append(float(t[3]))
        alpha_list.append(float(t[4]))
        alphaerror_list.append(float(t[5]))
        Transition_list.append(float(t[6])) 
        Transitionerror_list.append(float(t[7]))
        Time_list.append(float(t[8]))
        

    fp.close()
    
    mean = np.asarray(mean_list)
    meanerror = np.asarray(meanerror_list)
    sigma = np.asarray(sigma_list)
    sigmaerror = np.asarray(sigmaerror_list)
    alpha = np.asarray(alpha_list)
    alphaerror = np.asarray(alphaerror_list)
    Transition = np.asarray(Transition_list)
    Transitionerror = np.asarray(Transitionerror_list)
    Time = np.asarray(Time_list)

    return mean,meanerror, sigma,sigmaerror,alpha,alphaerror,Transition,Transitionerror,Time

def readparamterlognormal(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    mean_list = []  # units cm 
    sigma_list= []      # units ms-1
    
    meanerror_list = []  # units cm 
    sigmaerror_list= []      # units ms-1
    
    Time_list = []
    

    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        mean_list.append(float(t[0]))
        meanerror_list.append(float(t[1]))
        sigma_list.append(float(t[2]))
        sigmaerror_list.append(float(t[3]))  
        Time_list.append(float(t[4]))
        

    fp.close()
    
    mean = np.asarray(mean_list)
    meanerror = np.asarray(meanerror_list)
    sigma = np.asarray(sigma_list)
    sigmaerror = np.asarray(sigmaerror_list)   
    Time = np.asarray(Time_list)

    return mean,meanerror,sigma,sigmaerror,Time