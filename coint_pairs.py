# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:26:34 2018

Contains four fuctions

coint_time: is the main function that calls in the data, cleans it, runs the three test dataframes consisting of cointegrated pairs/triplets
over a certain time length
find_cointegrated_pairs
find_cointegrated_pairs_j
find_cointegrated_triplets

@author: Dell
"""
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import coint, adfuller
from CoJo import coint_johansen
from functions import *

def zscore(data):
    return (data-data.mean())/data.std()






def coint_time_cluster(start_date,end_date,zscore_level, pvalue,cluster):
    import datetime as dt
    from datetime import datetime
    df = pd.read_csv("ETFClust"+ repr(cluster)+".csv")
    df['date']=pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    col=df.iloc[0].isnull()
    col=col[col==True]
    col=col.index
    
    
    #years = years_back
    #days_per_year = 365.24
    #start_day = datetime.now() - dt.timedelta(days=(years*days_per_year))
    #start_day = start_day.date()
    
    start = datetime.strptime(start_date, '%Y/%m/%d')
    end = datetime.strptime(end_date, '%Y/%m/%d')
    
    df=df.loc[start.strftime("%Y/%m/%d"):]
    col=df.iloc[0].isnull()
    col=col[col==True]
    col=col.index
    data = df.drop(col.tolist(),axis=1)
    data.drop(data.columns[0],axis=1,inplace=True)
    data.to_csv("ETFClustClean"+ repr(cluster)+".csv")
    
    data_test=data.loc[start.strftime("%Y/%m/%d"):end.strftime("%Y/%m/%d")]
    data_test.to_csv("ETFClustTestClean"+ repr(cluster)+".csv")
    
    result_coint =  find_cointegrated_pairs(data_test,zscore_level, pvalue)
    result_jo =  find_cointegrated_pairs_j(data_test,zscore_level)
    result_trip = find_cointegrated_triplets(data, zscore_level)
    
    return result_coint, result_jo, result_trip

def coint_time_full_df(csv_file_name,start_date,end_date,zscore_level, pvalue):
    import datetime as dt
    from datetime import datetime
    df = pd.read_csv(csv_file_name+".csv")
    df['date']=pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    col=df.iloc[0].isnull()
    col=col[col==True]
    col=col.index
    
    
    #years = years_back
    #days_per_year = 365.24
    #start_day = datetime.now() - dt.timedelta(days=(years*days_per_year))
    #start_day = start_day.date()
    
    start = datetime.strptime(start_date, '%Y/%m/%d')
    end = datetime.strptime(end_date, '%Y/%m/%d')
    
    df=df.loc[start.strftime("%Y/%m/%d"):]
    col=df.iloc[0].isnull()
    col=col[col==True]
    col=col.index
    
    data = df.drop(col.tolist(),axis=1)
    #drop_list  = ["VIXY","GDXJ","SH","DXD","SPXU","TNA","TBT","SDOW","QID","SDS","TZA","SSO","DOG",
    #              "JNUG","JDST","SOXS","YINN"]
    #data.drop(drop_list,axis=1, inplace=True)
    
    data.drop(data.columns[0],axis=1,inplace=True)
    data.to_csv(csv_file_name+" Clean.csv")
    
    data_test=data.loc[start.strftime("%Y/%m/%d"):end.strftime("%Y/%m/%d")]
    data_test.to_csv(csv_file_name+ " Test Clean.csv")
    
    result_coint =  find_cointegrated_pairs(data_test,zscore_level, pvalue)
    result_jo =  find_cointegrated_pairs_j(data_test,zscore_level)
    #result_trip = find_cointegrated_triplets(data, zscore_level)
    
    return result_coint, result_jo


def find_cointegrated_pairs(data,zscore_level, pvalue_level):
    n = len(data.columns)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    b_coef = []
    #keys = securities.keys
    pairs = []
    df = pd.DataFrame(columns=["stk1_ind","stk2_dep", "beta", "pvalue","Hurst","HLife","Zscore","# crosses"])
    count=0
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[data.columns[i]]
            S2 = data[data.columns[j]]
            result = coint(S1, S2, trend="c")
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < pvalue_level:
                S1 = data[data.columns[i]]
                S2 = data[data.columns[j]]
                S1 = sm.add_constant(S1)
                results = sm.OLS(S2, S1).fit()
                S1 = data[data.columns[i]]
                b = results.params[1]
                if b>0:
                    spread = S2 - b * S1
                    adfuller=ts.adfuller(spread, 1, regression="c")[1]
                    
                    zs = zscore(spread)[-1]
                    if adfuller < 0.15:
                        if zs > zscore_level  or zs < -1.*zscore_level:                            
                            spread_norm= (spread-spread.mean()) /spread.std()
                            spread_norm=spread_norm[abs(spread_norm)>0.5]                            
                            count1=0
                            count_minus=-1                            
                            for k in range(0,len(spread_norm)-1):
                                if ((spread_norm[0]>0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                    count1=count1+1
                                
                                if ((spread_norm[0]<0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                    count_minus=count_minus+1

                            if (spread_norm[0]>0):
                                cross=count1
                            else:
                                cross=count_minus                          
                            pairs.append((data.columns[i], data.columns[j]))
                            b_coef.append((data.columns[i],b))
                            hurst_coef = round(hurst(spread),4)
                            Hlife = half_life(spread)
                            df.loc[count] = [data.columns[i], data.columns[j], b, pvalue, hurst_coef,Hlife,zs,cross]
                            #df.sort_values(by=["HLife"], inplace=True)
                            
                            count=count+1
                            
    columns = data.columns.tolist()
    columns = columns[::-1]
    data = data[columns]                    
    for i in range(0,n):
        for j in range(i+1, n):
            S1 = data[data.columns[i]]
            S2 = data[data.columns[j]]
            result = coint(S1, S2, trend="c")
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < pvalue_level:
                S1 = data[data.columns[i]]
                S2 = data[data.columns[j]]
                S1 = sm.add_constant(S1)
                results = sm.OLS(S2, S1).fit()
                S1 = data[data.columns[i]]
                b = results.params[1]
                if b>0:
                    spread = S2 - b * S1
                    adfuller=ts.adfuller(spread, 1, regression="c")[1]
                    
                    zs = zscore(spread)[-1]
                    if adfuller < 0.15:
                        if zs > zscore_level  or zs < -1.*zscore_level:                            
                            spread_norm= (spread-spread.mean()) /spread.std()
                            spread_norm=spread_norm[abs(spread_norm)>1]                            
                            count1=0
                            count_minus=-1                            
                            for k in range(0,len(spread_norm)-1):
                                if ((spread_norm[0]>0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                    count1=count1+1
                                
                                if ((spread_norm[0]<0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                    count_minus=count_minus+1

                            if (spread_norm[0]>0):
                                cross=count1
                            else:
                                cross=count_minus                          
                            pairs.append((data.columns[i], data.columns[j]))
                            b_coef.append((data.columns[i],b))
                            hurst_coef = round(hurst(spread),4)
                            Hlife = half_life(spread)
                            df.loc[count] = [data.columns[i], data.columns[j], b, pvalue, hurst_coef,Hlife,zs,cross]
                            df.sort_values(by=["HLife"], inplace=True)
                            
                        count=count+1   
    return  df

# =============================================================================
# dfu = find_cointegrated_pairs(data,1.5,0.05)
# =============================================================================
#import seaborn
#seaborn.heatmap(pvalues, xticklabels=data.columns, yticklabels=data.columns, cmap='RdYlGn_r' 
#                , mask = (pvalues >= 0.95)
#                )
# =============================================================================
# data=data_test
# index="XLB"
# zscore_level=0.2
# pvalue_level=0.5
# =============================================================================
def find_cointegrated_pairs_index(data,index,zscore_level, pvalue_level):
    n = len(data.columns)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    b_coef = []
    #keys = securities.keys
    pairs = []
    df = pd.DataFrame(columns=["stk1_ind","stk2_dep", "beta", "pvalue","Hurst","HLife","Zscore","# crosses"])
    count=0
 
    for j in range(n):
        S1 = data[index]
        S2 = data[data.columns[j]]
        result = coint(S1, S2, trend="c")
        score = result[0]
        pvalue = result[1]

        if pvalue < pvalue_level:
            S1 = data[index]
            S2 = data[data.columns[j]]
            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1).fit()
            S1 = data[index]
            b = results.params[1]
            if b>0:
                spread = S2 - b * S1
                adfuller=ts.adfuller(spread, 1, regression="c")[1]
                
                zs = zscore(spread)[-1]
                if adfuller < 0.15:
                    if zs > zscore_level  or zs < -1.*zscore_level:                            
                        spread_norm= (spread-spread.mean()) /spread.std()
                        spread_norm=spread_norm[abs(spread_norm)>1]                            
                        count1=0
                        count_minus=-1                            
                        for k in range(0,len(spread_norm)-1):
                            if ((spread_norm[0]>0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                count1=count1+1
                            
                            if ((spread_norm[0]<0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                count_minus=count_minus+1

                        if (spread_norm[0]>0):
                            cross=count1
                        else:
                            cross=count_minus                          
                        pairs.append((index, data.columns[j]))
                        b_coef.append((index,b))
                        hurst_coef = round(hurst(spread),4)
                        Hlife = half_life(spread)
                        df.loc[count] = [index, data.columns[j], b, pvalue, hurst_coef,Hlife,zs,cross]
                        #df.sort_values(by=["HLife"], inplace=True)
                        
                        count=count+1
                            
# =============================================================================
#     columns = data.columns.tolist()
#     columns = columns[::-1]
#     data = data[columns]                    
# =============================================================================
    
    for i in range(n):
        S1 = data[data.columns[i]]
        S2 = data[index]
        result = coint(S1, S2, trend="c")
        score = result[0]
        pvalue = result[1]
        if pvalue < pvalue_level:
            S1 = data[data.columns[i]]
            S2 = data[index]
            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1).fit()
            S1 = data[data.columns[i]]
            b = results.params[1]
            if b>0:
                spread = S2 - b * S1
                adfuller=ts.adfuller(spread, 1, regression="c")[1]
                
                zs = zscore(spread)[-1]
                if adfuller < 0.15:
                    if zs > zscore_level  or zs < -1.*zscore_level:                            
                        spread_norm= (spread-spread.mean()) /spread.std()
                        spread_norm=spread_norm[abs(spread_norm)>1]                            
                        count1=0
                        count_minus=-1                            
                        for k in range(0,len(spread_norm)-1):
                            if ((spread_norm[0]>0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                count1=count1+1
                            
                            if ((spread_norm[0]<0) and (spread_norm[k]*spread_norm[k+1]<0)):
                                count_minus=count_minus+1

                        if (spread_norm[0]>0):
                            cross=count1
                        else:
                            cross=count_minus                          
                        pairs.append((data.columns[i], index))
                        b_coef.append((data.columns[i],b))
                        hurst_coef = round(hurst(spread),4)
                        Hlife = half_life(spread)
                        df.loc[count] = [data.columns[i], index, b, pvalue, hurst_coef,Hlife,zs,cross]
                        #df.sort_values(by=["HLife"], inplace=True)
                        
                    count=count+1   
    return  df


def find_cointegrated_pairs_j(data, zscore_level):
    n = len(data.columns)
    score_matrix = np.zeros((n, n))
    count = 0
    dfj = pd.DataFrame(columns=["stk1","stk2","weight1","weight2","weight2a", "adfuller","Hurst","HLife","Zscore","# crosses"])
    #keys = securities.keys
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[data.columns[i]]
            S2 = data[data.columns[j]]
            df=S1.to_frame().join(S2.to_frame())
            df.dropna(inplace=True)
            results = coint_johansen(df,0,1)
            w = results.evec[:, 0]
            if (w[0]*w[1])<0:
                scoreT = results.lr1[1] - results.cvt[1,1]
                #scoreT1 = results.lr1[0] - results.cvt[0,0]
                scoreE = results.lr2[1] - results.cvm[1,1]
                #scoreE1 = results.lr2[0] - results.cvm[0,0]
                if (scoreT > 0.0 and  scoreE > 0.0 ) :
                    
                    yport = pd.DataFrame.sum(w*df, axis=1)
                    adfuller = ts.adfuller(yport, 1, regression="c")[1]
                    zs = zscore(yport)[-1]
                    HLife = int(half_life(yport))
                    if (HLife>0):    
                        if zs > zscore_level or zs < -1*zscore_level:
                            yport_norm=(yport-yport.mean())/yport.std()
                            yport_norm=yport_norm[abs(yport_norm)>1]  
                            count1=0
                            count_minus=-1
                            for k in range(0,len(yport_norm)-1):
                                if ((yport_norm[0]>0) and (yport_norm[k]*yport_norm[k+1]<0)):
                                    count1=count1+1
                                
                                if ((yport_norm[0]<0) and (yport_norm[k]*yport_norm[k+1]<0)):
                                    count_minus=count_minus+1

                            if (yport_norm[0]>0):
                                cross=count1
                            else:
                                cross=count_minus
                            
                            hurst_coef = round(hurst(yport),4)
                            
                            weight2a=(1/w[0])*w[1]*100
                            dfj.loc[count] = [data.columns[i], data.columns[j], w[0], w[1] ,weight2a, adfuller, hurst_coef, HLife,zs,cross]
                            
                            pairs.append((data.columns[i], data.columns[j]))
                            count=count+1
    dfj.sort_values(by=["HLife"], inplace=True)
    return dfj

def find_cointegrated_triplets(data, zscore_level):
    n = len(data.columns)
    score_matrix = np.zeros((n, n))
    count = 0
    dfj = pd.DataFrame(columns=["stk1","stk2","stk3","weight1","weight2","weight3","Hurst","HLife","Zscore","# crosses"])
    #keys = securities.keys
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1,n):
                S1 = data[data.columns[i]]
                S2 = data[data.columns[j]]
                S3 = data[data.columns[k]]
                df=S1.to_frame().join(S2.to_frame())
                df=df.join(S3)
                df.dropna(inplace=True)
                results = coint_johansen(df,0,1)
                
                #scoreT = results.lr1[1] - results.cvt[1,0]
                #scoreT1 = results.lr1[0] - results.cvt[0,0]
                scoreT2 = results.lr1[2] - results.cvt[2,0]
                #scoreE = results.lr2[1] - results.cvm[1,0]
                #scoreE1 = results.lr2[0] - results.cvm[0,0]
                scoreE2 = results.lr2[2] - results.cvm[2,0]
                
                if (scoreT2 >0.0 and scoreE2 >0.0):
                    w = results.evec[:, 0]
                    yport = pd.DataFrame.sum(w*df, axis=1)
                    zs = zscore(yport)[-1]
                    
                    if zs > zscore_level or zs < -1*zscore_level:
                        yport_norm=(yport-yport.mean())/yport.std()
                        yport_norm=yport_norm[abs(yport_norm)>1]  
                        count=0
                        count_minus=-1
                        for m in range(0,len(yport_norm)-1):
                            if ((yport_norm[0]>0) and (yport_norm[m]*yport_norm[m+1]<0)):
                                count=count+1
                            
                            if ((yport_norm[0]<0) and (yport_norm[m]*yport_norm[m+1]<0)):
                                count_minus=count_minus+1

                        if (yport_norm[0]>0):
                            cross=count
                        else:
                            cross=count_minus
                 
                        adfuller = ts.adfuller(yport, 1, regression="c")[1]
                        hurst_coef = round(hurst(yport),4)
                        Hlife = int(half_life(yport))
                        weight2a=(1/w[0])*w[1]*100
                        dfj.loc[count] = [data.columns[i], data.columns[j],data.columns[k], w[0], w[1], w[2] , hurst_coef, Hlife,zs,cross]
                        dfj.sort_values(by=["Hurst"], inplace=True)
                        pairs.append((data.columns[i], data.columns[j]))
                        count=count+1
    return dfj

    

  
    
