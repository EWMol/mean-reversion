# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:36:54 2018

@author: Dell
"""

#from IBUtil import *
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt


from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract as IBcontract
from threading import Thread
import queue
import datetime
import pandas as pd

DEFAULT_HISTORIC_DATA_ID=50
DEFAULT_GET_CONTRACT_ID=43

## marker for when queue is finished
FINISHED = object()
STARTED = object()
TIME_OUT = object()
 


#security_list = pd.read_csv("ETF List.csv",index_col=0)#.fillna("")
#historic_closen = pd.DataFrame(columns=["date"]).set_index("date")
#historic_close = pd.read_csv("historic_close.csv",index_col=0) # historic_close = pd.DataFrame(columns=["date"]).set_index("date")
#data_endtime = dt.now()
#ib_app = IBApp("127.0.0.1", 7497, 111)
#try:
#
#    start = datetime.datetime.now()
#    historic_close = ib_app.get_IB_historical_close(historic_closen,
#                                                    security_list,
#                                                    endDateTime=data_endtime,
#                                                    durationStr="13 Y",
#                                                    barSizeSetting="1 day")
#    print("Average %6.2f seconds per security" % ((datetime.datetime.now() - start).seconds / security_list.shape[0]))
#
#except Exception as e:
#    ib_app.disconnect()
#    raise
#ib_app.disconnect()
#
#historic_close.to_csv("ETF Data.csv")""""

# =============================================================================
# series=pd.read_csv("Utilities.csv")
# series=series["Symbol"].tolist()
# #series[146:]
# add_data_mean_rev(series, "2 Y", '1 day', 'Utilities Live Data.csv', 231,1)
# 
# =============================================================================



def download_data(csv_file_name,index):
    import math
    series=pd.read_csv(csv_file_name)
    series=series["Symbol"].tolist()
    if(len(series)>90):
        jump=0
        end_ser=90
        for i in range(math.ceil(len(series)/90)):
            df=add_data_mean_rev(series[jump:end_ser], "2 Y", '1 day', index+repr(i)+' Live Data.csv', 11,1)
            jump=jump+90
            end_ser=len(series)
            if(i==0):
                dfd=df
        dfd=pd.concat([dfd,df],axis=1)
        dfd.to_csv(index+" Live Data.csv")
    else:
        add_data_mean_rev(series, "2 Y", '1 day', index+' Live Data.csv', 11,1)
        
            


def add_data_mean_rev(series,lookback, bar_size, destination_file, id1,rth):
    
    security_list=pd.read_csv("securityTemplate.csv")
    for i in range(len(series)):
        security_list.iat[i,2]=series[i]
     
    security_list=security_list[security_list["symbol"] !="STK"]
    historic_closen = pd.DataFrame(columns=["date"]).set_index("date")
    
    data_endtime = dt.now()
    ib_app = IBAppMean("127.0.0.1", 7497, id1)
    try:
    
        start = datetime.datetime.now()
        historic_close = ib_app.get_IB_historical_close(historic_closen,
                                                        security_list,
                                                        endDateTime=data_endtime,
                                                        durationStr=lookback,
                                                        barSizeSetting=bar_size,
                                                        useRTH=rth)
        print("Average %6.2f seconds per security" % ((datetime.datetime.now() - start).seconds / security_list.shape[0]))
    
    except Exception as e:
        ib_app.disconnect()
        raise
    ib_app.disconnect()
    
    historic_close.to_csv(destination_file)
    return historic_close


def add_data(series,lookback, bar_size, destination_file, id1,rth):
    
    security_list=pd.read_csv("securityTemplate.csv")
    for i in range(len(series)):
        security_list.iat[i,2]=series[i]
     
    security_list=security_list[security_list["symbol"] !="STK"]
    historic_closen = pd.DataFrame(columns=["date"]).set_index("date")
    
    data_endtime = dt.now()
    ib_app = IBApp("127.0.0.1", 7497, id1)
    try:
    
        start = datetime.datetime.now()
        historic_close = ib_app.get_IB_historical_close(historic_closen,
                                                        security_list,
                                                        endDateTime=data_endtime,
                                                        durationStr=lookback,
                                                        barSizeSetting=bar_size,
                                                        useRTH=rth)
        print("Average %6.2f seconds per security" % ((datetime.datetime.now() - start).seconds / security_list.shape[0]))
    
    except Exception as e:
        ib_app.disconnect()
        raise
    ib_app.disconnect()
    
    historic_close.to_csv(destination_file)
    return historic_close

def add_data_single(stock,lookback, bar_size, destination_file, id1,rth):
    
    security_list = pd.DataFrame(columns=["secType", "symbol", "exchange", "currency", "lastTradeDateOrContractMonth"])
    security_list=security_list.append({"secType":"STK", "symbol":stock, "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
    historic_closen = pd.DataFrame(columns=["date"]).set_index("date")
    
    data_endtime = dt.now()
    ib_app = IBAppE("127.0.0.1", 7497, id1)
    try:
    
        start = datetime.datetime.now()
        historic_close = ib_app.get_IB_historical_close(historic_closen,
                                                        security_list,
                                                        endDateTime=data_endtime,
                                                        durationStr=lookback,
                                                        barSizeSetting=bar_size,
                                                        useRTH=rth)
        print("Average %6.2f seconds per security" % ((datetime.datetime.now() - start).seconds / security_list.shape[0]))
    
    except Exception as e:
        ib_app.disconnect()
        raise
    ib_app.disconnect()
    
    historic_close.to_csv(destination_file)
    return historic_close

#add_data_single("SPY", "1 Y", "1 day", "temp.csv", 121, 1)
#add_data(series,"4 Y", "1 day")



# Gist example of IB wrapper ...
#
# Download API from http://interactivebrokers.github.io/#
#
# Install python API code /IBJts/source/pythonclient $ python3 setup.py install
#
# Note: The test cases, and the documentation refer to a python package called IBApi,
#    but the actual package is called ibapi. Go figure.
#
# Get the latest version of the gateway:
# https://www.interactivebrokers.com/en/?f=%2Fen%2Fcontrol%2Fsystemstandalone-ibGateway.php%3Fos%3Dunix
#    (for unix: windows and mac users please find your own version)
#
# Run the gateway
#
# user: edemo
# pwd: demo123
#



class finishableQueue(object):

    def __init__(self, queue_to_finish):

        self._queue = queue_to_finish
        self.status = STARTED

    def get(self, timeout):
        """
        Returns a list of queue elements once timeout is finished, or a FINISHED flag is received in the queue
        :param timeout: how long to wait before giving up
        :return: list of queue elements
        """
        contents_of_queue=[]
        finished=False

        while not finished:
            try:
                current_element = self._queue.get(timeout=timeout)
                if current_element is FINISHED:
                    finished = True
                    self.status = FINISHED
                else:
                    contents_of_queue.append(current_element)
                    ## keep going and try and get more data

            except queue.Empty:
                ## If we hit a time out it's most probable we're not getting a finished element any time soon
                ## give up and return what we have
                finished = True
                self.status = TIME_OUT


        return contents_of_queue

    def timed_out(self):
        return self.status is TIME_OUT





class IBWrapper(EWrapper):
    """
    The wrapper deals with the action coming back from the IB gateway or TWS instance
    We override methods in EWrapper that will get called when this action happens, like currentTime
    Extra methods are added as we need to store the results in this object
    """

    def __init__(self):
        self._my_contract_details = {}
        self._my_historic_data_dict = {}
        self.init_error()

    ## error handling code
    def init_error(self):
        error_queue=queue.Queue()
        self._my_errors = error_queue

    def get_error(self, timeout=5):
        if self.is_error():
            try:
                return self._my_errors.get(timeout=timeout)
            except queue.Empty:
                return None

        return None

    def is_error(self):
        an_error_if=not self._my_errors.empty()
        return an_error_if

    def error(self, id, errorCode, errorString):
        ## Overriden method
        errormsg = "IB error id %d errorcode %d string %s" % (id, errorCode, errorString)
        self._my_errors.put(errormsg)


    ## get contract details code
    def init_contractdetails(self, reqId):
        contract_details_queue = self._my_contract_details[reqId] = queue.Queue()

        return contract_details_queue

    def contractDetails(self, reqId, contractDetails):
        ## overridden method

        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        ## overriden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(FINISHED)

    ## Historic data code
    def init_historicprices(self, tickerid):
        historic_data_queue = self._my_historic_data_dict[tickerid] = queue.Queue()

        return historic_data_queue


    def historicalData(self, tickerid , bar):

        ## Overriden method
        ## Note I'm choosing to ignore barCount, WAP and hasGaps but you could use them if you like
        bardata=(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume)
        
        historic_data_dict=self._my_historic_data_dict

        ## Add on to the current data
        if tickerid not in historic_data_dict.keys():
            self.init_historicprices(tickerid)

        historic_data_dict[tickerid].put(bardata)

    def historicalDataEnd(self, tickerid, start:str, end:str):
        ## overriden method

        if tickerid not in self._my_historic_data_dict.keys():
            self.init_historicprices(tickerid)

        self._my_historic_data_dict[tickerid].put(FINISHED)




class IBClient(EClient):
    """
    The client method
    We don't override native methods, but instead call them from our own wrappers
    """
    def __init__(self, wrapper):
        ## Set up with a wrapper inside
        EClient.__init__(self, wrapper)


    def resolve_ib_contract(self, ibcontract, reqId=DEFAULT_GET_CONTRACT_ID):

        """
        From a partially formed contract, returns a fully fledged version
        :returns fully resolved IB contract
        """

        ## Make a place to store the data we're going to return
        contract_details_queue = finishableQueue(self.init_contractdetails(reqId))

        print("Getting full contract details for " + ibcontract.symbol)

        self.reqContractDetails(reqId, ibcontract)

        ## Run until we get a valid contract(s) or get bored waiting
        MAX_WAIT_SECONDS = 10
        new_contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if contract_details_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        if len(new_contract_details)==0:
            print("Failed to get additional contract details: returning unresolved contract")
            return ibcontract

        if len(new_contract_details)>1:
            print("got multiple contracts using first one")

        new_contract_details=new_contract_details[0]

        resolved_ibcontract=new_contract_details.summary

        return resolved_ibcontract


    def get_IB_historical_data(self, ibcontract, 
                               endDateTime=datetime.datetime.today(),
                               durationStr="1 Y",
                               barSizeSetting="1 day",
                               whatToShow="Trades",
                               useRTH=0,
                               formatDate=1,
                               KeepUpToDate=False,
                               tickerid=DEFAULT_HISTORIC_DATA_ID
                               ):

        """
        Returns historical prices for a contract, up to today
        ibcontract is a Contract
        :returns list of prices in 4 tuples: Open high low close volume
        """


        ## Make a place to store the data we're going to return
        historic_data_queue = finishableQueue(self.init_historicprices(tickerid))

        # Request some historical data. Native method in EClient
        self.reqHistoricalData(
            tickerid,  # tickerId,
            ibcontract,  # contract,
            endDateTime.strftime("%Y%m%d %H:%M:%S %Z"),  # endDateTime,
            durationStr,  # durationStr,
            barSizeSetting,  # barSizeSetting,
            whatToShow,  # whatToShow,
            useRTH,  # useRTH,
            formatDate,  # formatDate
            KeepUpToDate,  # KeepUpToDate <<==== added for api 9.73.2
            [] ## chartoptions not used
        )



        ## Wait until we get a completed data, an error, or get bored waiting
        MAX_WAIT_SECONDS = 30
        print("Getting historical data for " +  ibcontract.symbol)

        historic_data = historic_data_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if historic_data_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        self.cancelHistoricalData(tickerid)


        return historic_data



class IBApp(IBWrapper, IBClient):
    def __init__(self, ipaddress, portid, clientid):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

        self.init_error()

    def get_IB_historical_close(self, historic_close,
                                security_list,
                                endDateTime=datetime.datetime.today(),
                                #endDateTime=dt.strptime("2015-12-30","%Y-%m-%d"),
                                durationStr="1 Y",
                                barSizeSetting="1 day",
                                whatToShow="Trades",
                                useRTH=0,
                                formatDate=1,
                                KeepUpToDate=False,
                                tickerid=DEFAULT_HISTORIC_DATA_ID
                                ):
        
            
        ibcontract = IBcontract()
        ibcontract.secType = "STK"
        ibcontract.symbol = "SPY"
        ibcontract.exchange = "SMART/ISLAND"
        ibcontract.currency = "USD"

        resolved_ibcontract = IBClient.resolve_ib_contract(self,ibcontract)
        historical_data = IBClient.get_IB_historical_data(self,
                                                        resolved_ibcontract,
                                                        endDateTime,
                                                        durationStr,
                                                        barSizeSetting,
                                                        whatToShow,
                                                        useRTH,
                                                        formatDate,
                                                        KeepUpToDate,
                                                        tickerid)
        
        main_index = pd.DataFrame(historical_data,columns = ["date","open","high","low","close","volume"])[:-1]
        main_index["date"] = pd.to_datetime(main_index['date'])
        main_index=main_index.set_index('date')
        main_index.drop(["open","high","low","close","volume"], inplace=True,axis=1)
        
        
        
        
        for index, security in security_list.iterrows():
            ibcontract = IBcontract()
            ibcontract.secType = security.secType
            ibcontract.lastTradeDateOrContractMonth = security.lastTradeDateOrContractMonth
            ibcontract.symbol = security.symbol
            ibcontract.exchange = security.exchange
            ibcontract.currency = security.currency
    
            resolved_ibcontract = IBClient.resolve_ib_contract(self,ibcontract)
            historical_data = IBClient.get_IB_historical_data(self,
                                                            resolved_ibcontract,
                                                            endDateTime,
                                                            durationStr,
                                                            barSizeSetting,
                                                            whatToShow,
                                                            useRTH,
                                                            formatDate,
                                                            KeepUpToDate,
                                                            tickerid)
           
            historical_df = pd.DataFrame(historical_data,columns = ["date","open","high","low","close","volume"])[:-1]
            historical_df["symbol"]=security.symbol
            historical_df["date"] = pd.to_datetime(historical_df['date'])
            historical_df=historical_df.set_index('date')
            historical_df=historical_df.combine_first(main_index)
            historical_df.reset_index(inplace=True)
            historical_df=historical_df.set_index(["symbol","date"])
            
            
            
            #historical_df["date"] = pd.to_datetime(historical_df['date'])
            #historical_df=historical_df.set_index('date')
            #historical_df.sort_index(inplace=True)
            historic_close = historic_close.append(historical_df)
        
        historic_close.reset_index(inplace=True)
        historic_close=historic_close.set_index(["symbol","date"])
        
        return historic_close
    
class IBAppMean(IBWrapper, IBClient):
    def __init__(self, ipaddress, portid, clientid):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

        self.init_error()

    def get_IB_historical_close(self, historic_close,
                                security_list,
                                endDateTime=datetime.datetime.today(),
                                durationStr="1 Y",
                                barSizeSetting="1 day",
                                whatToShow="Trades",
                                useRTH=0,
                                formatDate=1,
                                KeepUpToDate=False,
                                tickerid=DEFAULT_HISTORIC_DATA_ID
                                ):
        for index, security in security_list.iterrows():
            ibcontract = IBcontract()
            ibcontract.secType = security.secType
            ibcontract.lastTradeDateOrContractMonth = security.lastTradeDateOrContractMonth
            ibcontract.symbol = security.symbol
            ibcontract.exchange = security.exchange
            ibcontract.currency = security.currency
    
            resolved_ibcontract = IBClient.resolve_ib_contract(self,ibcontract)
            historical_data = IBClient.get_IB_historical_data(self,
                                                            resolved_ibcontract,
                                                            endDateTime,
                                                            durationStr,
                                                            barSizeSetting,
                                                            whatToShow,
                                                            useRTH,
                                                            formatDate,
                                                            KeepUpToDate,
                                                            tickerid)
           
            historical_df = pd.DataFrame(historical_data,columns = ["date","open","high","low","close","volume"])
            historical_df.head()
            historical_df = historical_df[["date", "close"]]
            historical_df.rename(columns={'close':security.symbol}, inplace=True)
            
            historical_df["date"] = pd.to_datetime(historical_df['date'])
            historical_df=historical_df.set_index('date')
            #historical_df.sort_index(inplace=True)
            historic_close = historical_df.combine_first(historic_close)
        return historic_close    
    

class IBAppE(IBWrapper, IBClient):
    def __init__(self, ipaddress, portid, clientid):
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

        self.init_error()

    def get_IB_historical_close(self, historic_close,
                                security_list,
                                endDateTime=datetime.datetime.today(),
                                durationStr="1 Y",
                                barSizeSetting="1 day",
                                whatToShow="Trades",
                                useRTH=0,
                                formatDate=1,
                                KeepUpToDate=False,
                                tickerid=DEFAULT_HISTORIC_DATA_ID
                                ):
        for index, security in security_list.iterrows():
            ibcontract = IBcontract()
            ibcontract.secType = security.secType
            ibcontract.lastTradeDateOrContractMonth = security.lastTradeDateOrContractMonth
            ibcontract.symbol = security.symbol
            ibcontract.exchange = security.exchange
            ibcontract.currency = security.currency
    
            resolved_ibcontract = IBClient.resolve_ib_contract(self,ibcontract)
            historical_data = IBClient.get_IB_historical_data(self,
                                                            resolved_ibcontract,
                                                            endDateTime,
                                                            durationStr,
                                                            barSizeSetting,
                                                            whatToShow,
                                                            useRTH,
                                                            formatDate,
                                                            KeepUpToDate,
                                                            tickerid)
           
            historical_df = pd.DataFrame(historical_data,columns = ["date","open","high","low","close","volume"])

            historical_df["date"] = pd.to_datetime(historical_df['date'])
            historical_df=historical_df.set_index('date')
            #historical_df.sort_index(inplace=True)
            historic_close = historical_df.combine_first(historic_close)
        return historic_close
    
  
#if __name__ == '__main__':

#app = IBApp("127.0.0.1", 7496, 15)
#
#security_list = pd.DataFrame(columns=["secType", "symbol", "exchange", "currency", "lastTradeDateOrContractMonth"])
#security_list=security_list.append({"secType":"STK", "symbol":"SPY", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"EEM", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"XLF", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"QQQ", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"VXX", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"UVXY", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"TVIX", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"GDX", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"EFA", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"FXI", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#security_list=security_list.append({"secType":"STK", "symbol":"IWM", "exchange":"SMART/ISLAND", "currency":"USD", "lastTradeDateOrContractMonth":""},ignore_index=True)
#
#historic_close = pd.DataFrame(columns=["date"]).set_index("date")
#
#try:
#    historic_close = app.get_IB_historical_close(historic_close,
#                                                security_list,
#                                                endDateTime=datetime.datetime.today(),
#                                                durationStr="1 W",
#                                                barSizeSetting="1 day")
#except Exception as e:
#    app.disconnect()
#    raise
#    print("Error: ", e.__doc__)
#app.disconnect()
#    
#historic_close.head()
