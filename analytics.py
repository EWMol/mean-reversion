# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:54:56 2018

@author: Dell
"""
import pairs_time
from numpy.matlib import repmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import statsmodels.tsa.stattools as ts
#dfc4, dfj4 = coint_time("2017/4/23","2018/2/2",1.5,0.10)





def plot_pairs_clust(data, start, end, days_forward, moving_average_fast, moving_average_slow,cluster):
    import datetime as dt
    from datetime import datetime
    start_date = start
    end_date = end
    end1 = datetime.strptime(end, '%Y/%m/%d')
    end_plus = end1 + dt.timedelta(days=days_forward)
    end_plus = end_plus.strftime("%Y/%m/%d") 
    end_minus = end1 - dt.timedelta(days=1)
    end_minus = end_minus.strftime("%Y/%m/%d")

    df = pd.read_csv("ETFClustClean"+ repr(cluster)+".csv")
    df['date']=pd.to_datetime(df['date'],yearfirst=True, format="%Y/%m/%d")
    df.set_index('date', inplace=True)
    
    for i in range(0,len(data)):
        dfc2=data
        stk_ind=dfc2.iloc[i][0]
        stk_dep=dfc2.iloc[i][1]
        beta =dfc2.iloc[i][2]
        
        spread = df[stk_dep].loc[start_date:end_plus] - beta * df[stk_ind].loc[start_date:end_plus]
        spread=spread+10000
        adf=[0]*3
        for  i in range(3,len(spread)):
            adf.append(ts.adfuller(spread.loc[:spread.index[i].strftime("%Y/%m/%d")], 1, regression="c")[1])    
            
        dat = df[[stk_ind,stk_dep]].loc[start_date:end_plus]
        dat["adf"]=adf
        dat["port"] = spread
        
        dat["slow_ma"] = dat["port"].rolling(moving_average_fast).mean()
        dat["fast_ma"] = dat["port"].rolling(moving_average_slow).mean()
        
        mean = dat["port"].loc[start_date:end_date].mean()
        std = dat["port"].loc[start_date:end_date].std()
        
        dat2=dat.loc[start_date:end_date]
        dat2["zscore"]=(dat2["port"]-dat2["port"].mean())/dat2["port"].std()
        dat.dropna(inplace=True)
        
        print([stk_ind,stk_dep])
        print(dat2["adf"][-1])
        #print(spread)

        #plt.plot(spread.index,spread)
        #xopt1 = calibrate(spread[0:699])
        #S1 = generate_sims(spread[698],xopt1,100,10000)
        #plt.plot(spread.index[-100:],S1)
        
        #plt.plot(spread.index[-100:],spread[-100:], color='white')
        #plt.show() 

        spread1=spread.loc[end_minus:]


        f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 20))        

        ax1.plot(dat2["zscore"])
        ax1.axhline(0, color='black')
        ax1.axhline(-1, color='green', linestyle='--')
        ax1.axhline(1, color='green', linestyle='--')
        ax1.axhline(2, color='orange', linestyle='--')
        ax1.axhline(-2, color='orange', linestyle='--')
        ax1.axhline(3,color='red', linestyle='--')
        ax1.axhline(-3, color='red', linestyle='--')    
        ax1.legend([str(stk_ind),str(stk_dep)])

        ax2.plot(spread.index,spread)
        xopt1 = calibrate(spread.loc[start_date:end_date])
        S1 = generate_sims(spread.loc[end_minus],xopt1,len(spread1),500)
        ax2.plot(spread1.index,S1)
        ax2.plot(spread1.index,spread.loc[end_minus:], color='black',linewidth=2.0)
        #ax2.plot(dat["port"],label=[stk_ind,stk_dep])
        #ax2.plot(dat["slow_ma"],color='pink')
        #ax2.plot(dat["fast_ma"],color='pink')
        #ax2.axhline(dat["port"].loc[end_date], color='purple')            
        ax2.axhline(mean, color='black')
        ax2.axhline(mean + std, color='green', linestyle='--')
        ax2.axhline(mean - std, color='green', linestyle='--')
        ax2.axhline(mean + 2*std, color='orange', linestyle='--')
        ax2.axhline(mean - 2*std, color='orange', linestyle='--')
        ax2.axhline(mean + 3*std, color='red', linestyle='--')
        ax2.axhline(mean - 3*std, color='red', linestyle='--')        
        ax2.axvline(pd.to_datetime(end), color='purple')
        ax2.legend([str(stk_ind),str(stk_dep)])
        ax3.plot(dat["adf"],label=[stk_ind,stk_dep])
        ax3.axvline(pd.to_datetime(end), color='purple')
        ax3.legend([str(stk_ind),str(stk_dep)])

def plot_pairs(data, csv_file_name, start, end, days_forward, moving_average_fast, moving_average_slow):
    import datetime as dt
    from datetime import datetime
    start_date = start
    end_date = end
    end1 = datetime.strptime(end, '%Y/%m/%d')
    end_plus = end1 + dt.timedelta(days=days_forward)
    end_plus = end_plus.strftime("%Y/%m/%d") 
    end_minus = end1 - dt.timedelta(days=1)
    end_minus = end_minus.strftime("%Y/%m/%d")

    df = pd.read_csv(csv_file_name + " Data Clean.csv")
    df['date']=pd.to_datetime(df['date'],yearfirst=True, format="%Y/%m/%d")
    df.set_index('date', inplace=True)
    
    for i in range(0,len(data)):
        dfc2=data
        stk_ind=dfc2.iloc[i][0]
        stk_dep=dfc2.iloc[i][1]
        beta =dfc2.iloc[i][2]
        
        spread = df[stk_dep].loc[start_date:end_plus] - beta * df[stk_ind].loc[start_date:end_plus]
        spread=spread+10000
        adf=[0]*3
        for  i in range(3,len(spread)):
            adf.append(ts.adfuller(spread.loc[:spread.index[i].strftime("%Y/%m/%d")], 1, regression="c")[1])    
            
        dat = df[[stk_ind,stk_dep]].loc[start_date:end_plus]
        dat["adf"]=adf
        dat["port"] = spread
        
        dat["slow_ma"] = dat["port"].rolling(moving_average_fast).mean()
        dat["fast_ma"] = dat["port"].rolling(moving_average_slow).mean()
        
        mean = dat["port"].loc[start_date:end_date].mean()
        std = dat["port"].loc[start_date:end_date].std()
        
        dat2=dat.loc[start_date:end_date]
        dat2["zscore"]=(dat2["port"]-dat2["port"].mean())/dat2["port"].std()
        dat.dropna(inplace=True)
        
        print([stk_ind,stk_dep])
        print(dat2["adf"][-1])
        #print(spread)

        #plt.plot(spread.index,spread)
        #xopt1 = calibrate(spread[0:699])
        #S1 = generate_sims(spread[698],xopt1,100,10000)
        #plt.plot(spread.index[-100:],S1)
        
        #plt.plot(spread.index[-100:],spread[-100:], color='white')
        #plt.show() 

        spread1=spread.loc[end_minus:]


        f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 20))        

        ax1.plot(dat2["zscore"])
        ax1.axhline(0, color='black')
        ax1.axhline(-1, color='green', linestyle='--')
        ax1.axhline(1, color='green', linestyle='--')
        ax1.axhline(2, color='orange', linestyle='--')
        ax1.axhline(-2, color='orange', linestyle='--')
        ax1.axhline(3,color='red', linestyle='--')
        ax1.axhline(-3, color='red', linestyle='--')    
        ax1.legend([str(stk_ind),str(stk_dep)])

        ax2.plot(spread.index,spread)
        #xopt1 = calibrate(spread.loc[start_date:end_date])
        #S1 = generate_sims(spread.loc[end_minus],xopt1,len(spread1),500)
        #ax2.plot(spread1.index,S1)
        #ax2.plot(spread1.index,spread.loc[end_minus:], color='black',linewidth=2.0)
        #ax2.plot(dat["port"],label=[stk_ind,stk_dep])
        #ax2.plot(dat["slow_ma"],color='pink')
        #ax2.plot(dat["fast_ma"],color='pink')
        #ax2.axhline(dat["port"].loc[end_date], color='purple')            
        ax2.axhline(mean, color='black')
        ax2.axhline(mean + std, color='green', linestyle='--')
        ax2.axhline(mean - std, color='green', linestyle='--')
        ax2.axhline(mean + 2*std, color='orange', linestyle='--')
        ax2.axhline(mean - 2*std, color='orange', linestyle='--')
        ax2.axhline(mean + 3*std, color='red', linestyle='--')
        ax2.axhline(mean - 3*std, color='red', linestyle='--')        
        ax2.axvline(pd.to_datetime(end), color='purple')
        ax2.legend([str(stk_ind),str(stk_dep)])
        ax3.plot(dat["adf"],label=[stk_ind,stk_dep])
        ax3.axvline(pd.to_datetime(end), color='purple')
        ax3.legend([str(stk_ind),str(stk_dep)])



def plot_pairs_j(data,csv_file_name,start,end,days_forward,moving_average_fast,moving_average_slow):
    import datetime as dt
    from datetime import datetime
    start_date = start
    end_date = end
    end1 = datetime.strptime(end, '%Y/%m/%d')
    end_plus = end1 + dt.timedelta(days=days_forward)
    end_plus = end_plus.strftime("%Y/%m/%d") 
    
    
    df = pd.read_csv(csv_file_name + " Data Clean.csv")
    df['date']=pd.to_datetime(df['date'],yearfirst=True, format="%Y/%m/%d")
    df.set_index('date', inplace=True)
    
    for i in range(0,len(data)):
        dfj2=data
        stk1=dfj2.iloc[i][0]
        stk2=dfj2.iloc[i][1]
        weight1 =dfj2.iloc[i][2]
        weight2 =dfj2.iloc[i][3]
        
        spread = df[stk1].loc[start_date:end_plus]*weight1 + df[stk2].loc[start_date:end_plus]*weight2

        adf=[0]*3
        for  i in range(3,len(spread)):
            adf.append(ts.adfuller(spread.loc[:spread.index[i].strftime("%Y/%m/%d")], 1, regression="c")[1])            

        dat = df[[stk1,stk2]].loc[start_date:end_plus]
        dat["adf"]=adf
        dat["port"] = spread
        
        dat["slow_ma"] = dat["port"].rolling(moving_average_fast).mean()
        dat["fast_ma"] = dat["port"].rolling(moving_average_slow).mean()
        
        mean = dat["port"].loc[start_date:end_date].mean()
        std = dat["port"].loc[start_date:end_date].std()
        
        dat2=dat.loc[start_date:end_date]
        dat2["zscore"]=(dat2["port"]-dat2["port"].mean())/dat2["port"].std()
        dat.dropna(inplace=True)
      
        print([stk1,stk2])
        print(dat2["adf"][-1])
       
        f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 20))        

        ax1.plot(dat2["zscore"],label=[stk1,stk2])
        ax1.axhline(0, color='black')
        ax1.axhline(-1, color='green', linestyle='--')
        ax1.axhline(1, color='green', linestyle='--')
        ax1.axhline(2, color='orange', linestyle='--')
        ax1.axhline(-2, color='orange', linestyle='--')
        ax1.axhline(3,color='red', linestyle='--')
        ax1.axhline(-3, color='red', linestyle='--')        

        ax2.plot(dat["port"],label=[stk1,stk2])
        ax2.plot(dat["slow_ma"],color='pink')
        ax2.plot(dat["fast_ma"],color='pink')
        ax2.axhline(dat["port"].loc[end_date], color='purple')            
        ax2.axhline(mean, color='black')
        ax2.axhline(mean + std, color='green', linestyle='--')
        ax2.axhline(mean - std, color='green', linestyle='--')
        ax2.axhline(mean + 2*std, color='orange', linestyle='--')
        ax2.axhline(mean - 2*std, color='orange', linestyle='--')
        ax2.axhline(mean + 3*std, color='red', linestyle='--')
        ax2.axhline(mean - 3*std, color='red', linestyle='--')        
        ax2.axvline(pd.to_datetime(end), color='purple')

        ax3.plot(dat["adf"],label=[stk1,stk2])
        ax3.axvline(pd.to_datetime(end), color='purple')        
        

def plot_trips_j(data,csv_file_name,start,end,days_forward,moving_average_fast,moving_average_slow):
    import datetime as dt
    from datetime import datetime
    start_date = start
    end_date = end
    end1 = datetime.strptime(end, '%Y/%m/%d')
    end_plus = end1 + dt.timedelta(days=days_forward)
    end_plus = end_plus.strftime("%Y/%m/%d") 
    
    
    df = pd.read_csv(csv_file_name + " Data Clean.csv")
    df['date']=pd.to_datetime(df['date'],yearfirst=True, format="%Y/%m/%d")
    df.set_index('date', inplace=True)
    
    for i in range(0,len(data)):
        dfj2=data
        stk1=dfj2.iloc[i][0]
        stk2=dfj2.iloc[i][1]
        stk3=dfj2.iloc[i][2]
        weight1 =dfj2.iloc[i][3]
        weight2 =dfj2.iloc[i][4]
        weight3 =dfj2.iloc[i][5]
        
        
        spread = df[stk1].loc[start_date:end_plus]*weight1 + df[stk2].loc[start_date:end_plus]*weight2+ df[stk3].loc[start_date:end_plus]*weight3

        adf=[0]*3
        for  i in range(3,len(spread)):
            adf.append(ts.adfuller(spread.loc[:spread.index[i].strftime("%Y/%m/%d")], 1, regression="c")[1])            

        dat = df[[stk1,stk2,stk3]].loc[start_date:end_plus]
        dat["adf"]=adf
        dat["port"] = spread
        
        dat["slow_ma"] = dat["port"].rolling(moving_average_fast).mean()
        dat["fast_ma"] = dat["port"].rolling(moving_average_slow).mean()
        
        mean = dat["port"].loc[start_date:end_date].mean()
        std = dat["port"].loc[start_date:end_date].std()
        
        dat2=dat.loc[start_date:end_date]
        dat2["zscore"]=(dat2["port"]-dat2["port"].mean())/dat2["port"].std()
        dat.dropna(inplace=True)
      
        print([stk1,stk2,stk3])
        print(dat2["adf"][-1])
       
        f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 20))        

        ax1.plot(dat2["zscore"],label=[stk1,stk2,stk3])
        ax1.axhline(0, color='black')
        ax1.axhline(-1, color='green', linestyle='--')
        ax1.axhline(1, color='green', linestyle='--')
        ax1.axhline(2, color='orange', linestyle='--')
        ax1.axhline(-2, color='orange', linestyle='--')
        ax1.axhline(3,color='red', linestyle='--')
        ax1.axhline(-3, color='red', linestyle='--')        

        ax2.plot(dat["port"],label=[stk1,stk2,stk3])
        ax2.plot(dat["slow_ma"],color='pink')
        ax2.plot(dat["fast_ma"],color='pink')
        ax2.axhline(dat["port"].loc[end_date], color='purple')            
        ax2.axhline(mean, color='black')
        ax2.axhline(mean + std, color='green', linestyle='--')
        ax2.axhline(mean - std, color='green', linestyle='--')
        ax2.axhline(mean + 2*std, color='orange', linestyle='--')
        ax2.axhline(mean - 2*std, color='orange', linestyle='--')
        ax2.axhline(mean + 3*std, color='red', linestyle='--')
        ax2.axhline(mean - 3*std, color='red', linestyle='--')        
        ax2.axvline(pd.to_datetime(end), color='purple')

        ax3.plot(dat["adf"],label=[stk1,stk2])
        ax3.axvline(pd.to_datetime(end), color='purple')   



 

def calibrate(data):
    dt = 1/252. #daily    
    x=(data-data.mean())/data.std()
    
    initialParams = [.5, x.mean()-.5*np.std(x)**2, np.std(x)]
    loss_fun = lambda P:-np.sum(np.log((1/sqrt(2*pi*P[2]**2))*exp(-0.5*(np.diff(x)[1:] - P[0]*(P[1] - x[:-2]))**2/P[2]**2)))
    xopt = scipy.optimize.fmin(func=loss_fun, x0=initialParams)

    print("alpha: " + str(xopt[0]/dt))
    print("mu: " + str(xopt[1])) 
    print("sigma: " + str(xopt[2]/sqrt(dt)))
    return [xopt[0]/dt, xopt[1], xopt[2]/sqrt(dt)]

 

 

def generate_sims(initial_value,params,steps,nsims=500):
    
    dt = 1/252.
    alpha = params[0]
    mu = params[1]
    sigma = params[2]
    E = np.random.randn(steps,nsims)
    logS = np.ones((steps,nsims))*np.log(initial_value)
    for i in range(0,steps-1):
       logS[i+1] = logS[i]*exp(-alpha*dt) + mu*(1-exp(-alpha*dt)) + sqrt((sigma**2)*(1-exp(-2*alpha*dt))/(2*alpha))*E[i]
    simulations = exp(logS)
    return simulations
