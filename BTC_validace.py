'''
Created on 5. 3. 2018

@author: adam
'''

import numpy
from numpy import array
import talib
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from keras.models import load_model
from math import sqrt
import os
import warnings
from keras import backend
import plotly
from plotly.graph_objs import Scatter
import plotly.graph_objs as go

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") 		#Hide messy Numpy warnings

# transform series into train and test sets for supervised learning
def prepare_data(series, n_lag, n_seq, rsi_14, mfi_14, mom_14, plus_dm_14, dx_14, cci_14, aaron_14, cmo_14, roc_14, rorc_14, will_14, bop, ad, obv, avg_price, med_price, typ_price, wcl_price, atr_14, natr_14, trange, mid_price):
    
    # rescale values to 0, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    
    values_close = series['Close'].values
    #values_close_max = max(values_close)*2.5
    values_close_max = 20000
    values_close = numpy.append([values_close_max,0], values_close)	# add boundaries 
    values_close = values_close.reshape(len(values_close), 1) 
    scaled_values_close = scaler_close.fit_transform(values_close) 	# apply scaling 
    scaled_values_close = numpy.delete(scaled_values_close,[0,1])		# remove boundaries
    
    values_volume = series['Volume'].values
    #values_volume_max = max(values_volume)*2.5
    values_volume_max = 20000
    values_volume = numpy.append([values_volume_max,0], values_volume)	# add boundaries 
    values_volume = values_volume.reshape(len(values_volume), 1) 
    scaled_values_volume = scaler.fit_transform(values_volume) 			# apply scaling 
    scaled_values_volume = numpy.delete(scaled_values_volume,[0,1]) 		# remove boundaries
    
    values_rsi = rsi_14.reshape(len(scaled_values_close), 1)
    scaled_values_rsi = scaler.fit_transform(values_rsi)

    values_mfi = mfi_14.reshape(len(scaled_values_close), 1)
    scaled_values_mfi = scaler.fit_transform(values_mfi)

    values_mom = mom_14.reshape(len(scaled_values_close), 1)
    scaled_values_mom = scaler.fit_transform(values_mom)

    values_plus_dm = plus_dm_14.reshape(len(scaled_values_close), 1)
    scaled_values_plus_dm = scaler.fit_transform(values_plus_dm)

    values_dx = dx_14.reshape(len(scaled_values_close), 1)
    scaled_values_dx = scaler.fit_transform(values_dx)

    values_cci = cci_14.reshape(len(scaled_values_close), 1)
    scaled_values_cci = scaler.fit_transform(values_cci) 

    values_aaron = aaron_14.reshape(len(scaled_values_close), 1)
    scaled_values_aaron = scaler.fit_transform(values_aaron) 

    values_cmo = cmo_14.reshape(len(scaled_values_close), 1)
    scaled_values_cmo = scaler.fit_transform(values_cmo)

    values_roc = roc_14.reshape(len(scaled_values_close), 1)
    scaled_values_roc = scaler.fit_transform(values_roc)

    values_rorc = rorc_14.reshape(len(scaled_values_close), 1)
    scaled_values_rorc = scaler.fit_transform(values_rorc)

    values_will = will_14.reshape(len(scaled_values_close), 1)
    scaled_values_will = scaler.fit_transform(values_will)

    values_bop = bop.reshape(len(scaled_values_close), 1)
    scaled_values_bop = scaler.fit_transform(values_bop)

    values_ad = ad.reshape(len(scaled_values_close), 1)
    scaled_values_ad = scaler.fit_transform(values_ad)

    values_obv = obv.reshape(len(scaled_values_close), 1)
    scaled_values_obv = scaler.fit_transform(values_obv)

    values_avg_price = avg_price.reshape(len(scaled_values_close), 1)
    scaled_values_avg_price = scaler.fit_transform(values_avg_price)

    values_med_price = med_price.reshape(len(scaled_values_close), 1)
    scaled_values_med_price = scaler.fit_transform(values_med_price)

    values_typ_price = typ_price.reshape(len(scaled_values_close), 1)
    scaled_values_typ_price = scaler.fit_transform(values_typ_price)

    values_wcl_price = wcl_price.reshape(len(scaled_values_close), 1)
    scaled_values_wcl_price = scaler.fit_transform(values_wcl_price)

    values_atr_14 = atr_14.reshape(len(scaled_values_close), 1)
    scaled_values_atr_14 = scaler.fit_transform(values_atr_14)

    values_natr_14 = natr_14.reshape(len(scaled_values_close), 1)
    scaled_values_natr_14 = scaler.fit_transform(values_natr_14)

    values_trange = trange.reshape(len(scaled_values_close), 1)
    scaled_values_trange = scaler.fit_transform(values_trange)

    values_mid_price = mid_price.reshape(len(scaled_values_close), 1)
    scaled_values_mid_price = scaler.fit_transform(values_mid_price)

    # transform data to be stationary
    #--------------------------------------------
    diff_series_close = difference(scaled_values_close, 1)
    #diff_values_close = diff_series_close.values.reshape(len(diff_series_close), 1)
    diff_values_close = diff_series_close.values.reshape(len(diff_series_close), 1)

    diff_series_volume = difference(scaled_values_volume, 1)
    diff_values_volume = diff_series_volume.values.reshape(len(diff_series_volume), 1)

    diff_series_rsi = difference(scaled_values_rsi, 1)
    diff_values_rsi = diff_series_rsi.values.reshape(len(diff_series_rsi), 1)

    diff_series_mfi = difference(scaled_values_mfi, 1)
    diff_values_mfi = diff_series_mfi.values.reshape(len(diff_series_mfi), 1)

    diff_series_mom = difference(scaled_values_mom, 1)
    diff_values_mom = diff_series_mom.values.reshape(len(diff_series_mom), 1)

    diff_series_plus_dm = difference(scaled_values_plus_dm, 1)
    diff_values_plus_dm = diff_series_plus_dm.values.reshape(len(diff_series_plus_dm), 1)

    diff_series_dx = difference(scaled_values_dx, 1)
    diff_values_dx = diff_series_dx.values.reshape(len(diff_series_dx), 1)

    diff_series_cci = difference(scaled_values_cci, 1)
    diff_values_cci = diff_series_cci.values.reshape(len(diff_series_cci), 1) 

    diff_series_aaron = difference(scaled_values_aaron, 1)
    diff_values_aaron = diff_series_aaron.values.reshape(len(diff_series_aaron), 1) 

    diff_series_cmo = difference(scaled_values_cmo, 1)
    diff_values_cmo = diff_series_cmo.values.reshape(len(diff_series_cmo), 1)

    diff_series_roc = difference(scaled_values_roc, 1)
    diff_values_roc = diff_series_roc.values.reshape(len(diff_series_roc), 1)

    diff_series_rorc = difference(scaled_values_rorc, 1)
    diff_values_rorc = diff_series_rorc.values.reshape(len(diff_series_rorc), 1)

    diff_series_will = difference(scaled_values_will, 1)
    diff_values_will = diff_series_will.values.reshape(len(diff_series_will), 1)

    diff_series_bop = difference(scaled_values_bop, 1)
    diff_values_bop = diff_series_bop.values.reshape(len(diff_series_bop), 1)

    diff_series_ad = difference(scaled_values_ad, 1)
    diff_values_ad = diff_series_ad.values.reshape(len(diff_series_ad), 1)

    diff_series_obv = difference(scaled_values_obv, 1)
    diff_values_obv = diff_series_obv.values.reshape(len(diff_series_obv), 1)

    diff_series_avg_price = difference(scaled_values_avg_price, 1)
    diff_values_avg_price = diff_series_avg_price.values.reshape(len(diff_series_avg_price), 1)

    diff_series_med_price = difference(scaled_values_med_price, 1)
    diff_values_med_price = diff_series_med_price.values.reshape(len(diff_series_med_price), 1)

    diff_series_typ_price = difference(scaled_values_typ_price, 1)
    diff_values_typ_price = diff_series_typ_price.values.reshape(len(diff_series_typ_price), 1)

    diff_series_wcl_price = difference(scaled_values_wcl_price, 1)
    diff_values_wcl_price = diff_series_wcl_price.values.reshape(len(diff_series_wcl_price), 1)

    diff_series_trange = difference(scaled_values_trange, 1)
    diff_values_trange = diff_series_trange.values.reshape(len(diff_series_trange), 1)

    diff_series_natr_14 = difference(scaled_values_natr_14, 1)
    diff_values_natr_14 = diff_series_natr_14.values.reshape(len(diff_series_natr_14), 1)

    diff_series_atr_14 = difference(scaled_values_atr_14, 1)
    diff_values_atr_14 = diff_series_atr_14.values.reshape(len(diff_series_atr_14), 1)

    diff_series_mid_price = difference(scaled_values_mid_price, 1)
    diff_values_mid_price = diff_series_mid_price.values.reshape(len(diff_series_mid_price), 1)
    #--------------------------------------------
    together = numpy.concatenate((diff_values_mid_price, diff_values_atr_14, diff_values_natr_14, diff_values_trange, diff_values_wcl_price, diff_values_typ_price, diff_values_med_price, diff_values_avg_price, diff_values_obv, diff_values_ad, diff_values_bop, diff_values_will, diff_values_rorc, diff_values_roc, diff_values_cmo, diff_values_aaron, diff_values_cci, diff_values_dx, diff_values_plus_dm, diff_values_mom, diff_values_mfi, diff_values_volume, diff_values_rsi , diff_values_close), axis=1)
    
    #--------------------------------------------
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(together, n_lag, n_seq)
    test = supervised.values
    
    return scaler_close, test

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i).iloc[:,-1])
        if i == 0:
            names += ['var(t)']
        else:
            names += ['var(t+%d)' % i]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
def forecast_lstm(model, X, n_batch, n_lag):
    # reshape input pattern to [samples, timesteps, features]	
    X = X.reshape(n_batch, n_lag, n_features)
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_seq, n_lag):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, :-n_seq]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch, n_lag)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(price_series, forecasts, scaler_close):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler_close.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing    
        index = (len(price_series) - len(forecasts)-n_seq) + i
        last_ob = price_series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step

# plot the forecasts in the context of the original dataset
def plot_forecasts(price_series, forecasts):
    pyplot.figure()
    # plot the entire dataset in blue
    pyplot.plot(price_series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = (len(price_series) - len(forecasts)-n_seq) + i 
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [price_series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')	
        #print(off_s,off_e,len(price_series),len(forecasts))	

    # show the plot
    pyplot.show()

# configure


n_lag = 5		 # okno (z kolika předchozích hodnot)
n_seq = 3		 # počet predikovaných kroků t+n
n_features = 24   # Close, Volume, RSI..
n_batch = 1
rsi_n_day = 14
window = 4
price_window = 10  # 0-9,  i+6 = t
basic_model_data = 236

buy_allowance = True 
sell_allowance = False 

profit_actual = 0
profit_long = 0
price_mem = 0

operation = []
trades = []
date_remember = []
price_remember = []

par1 = 5 # UP %
par2 = 5 # DROP %

load_data = read_csv('data/csv/XXBTZEUR_2017_full.csv', header=0)
                #306
for k in range(1,307):
    print("====== zacatek %d iterace ======" %k)
    l = k*4  
    # load data
    load_data_window = load_data[:basic_model_data+l+n_seq]
    # load model
    backend.clear_session()
    model = load_model("models/BTC/test_2017_retrained_%d.h5" % (k-1))
    rsi_14 = talib.RSI(numpy.array(load_data_window['Close']), rsi_n_day)  	# Vypocet RSI //+14 hodnot
    rsi_14 = rsi_14[rsi_n_day:]										        # Nan value trim

    mfi_14 = talib.MFI(numpy.array(load_data_window['Max']),numpy.array(load_data_window['Min']),numpy.array(load_data_window['Close']),numpy.array(load_data_window['Volume']), rsi_n_day)
    mfi_14 = mfi_14[rsi_n_day:]	

    mom_14 = talib.MOM(numpy.array(load_data_window['Close']), rsi_n_day)
    mom_14 = mom_14[rsi_n_day:]

    plus_dm_14 = talib.PLUS_DM(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), rsi_n_day)	
    plus_dm_14 = plus_dm_14[rsi_n_day:]

    dx_14 = talib.DX(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), rsi_n_day)
    dx_14 = dx_14[rsi_n_day:]

    cci_14 = talib.CCI(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), rsi_n_day)
    cci_14 = cci_14[rsi_n_day:]

    aaron_14 = talib.AROONOSC(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), rsi_n_day)
    aaron_14 = aaron_14[rsi_n_day:]

    cmo_14 = talib.CMO(numpy.array(load_data_window['Close']), rsi_n_day)
    cmo_14 = cmo_14[rsi_n_day:]

    roc_14 = talib.ROC(numpy.array(load_data_window['Close']), rsi_n_day)
    roc_14 = roc_14[rsi_n_day:]

    rorc_14 = talib.ROCR(numpy.array(load_data_window['Close']), rsi_n_day)
    rorc_14 = rorc_14[rsi_n_day:]

    will_14 = talib.WILLR(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), rsi_n_day)
    will_14 = will_14[rsi_n_day:]

    bop = talib.BOP(numpy.array(load_data_window['Open']), numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']))
    bop = bop[rsi_n_day:]

    ad = talib.AD(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), numpy.array(load_data_window['Volume']))
    ad = ad[rsi_n_day:]

    obv = talib.OBV(numpy.array(load_data_window['Close']), numpy.array(load_data_window['Volume']))
    obv = obv[rsi_n_day:]

    avg_price = talib.AVGPRICE(numpy.array(load_data_window['Open']), numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']))
    avg_price = avg_price[rsi_n_day:]

    med_price = talib.MEDPRICE(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']))
    med_price = med_price[rsi_n_day:]

    typ_price = talib.TYPPRICE(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']))
    typ_price = typ_price[rsi_n_day:]

    wcl_price = talib.WCLPRICE(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']))
    wcl_price = wcl_price[rsi_n_day:]

    atr_14 = talib.ATR(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), rsi_n_day)
    atr_14 = atr_14[rsi_n_day:]

    natr_14 = talib.NATR(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']), rsi_n_day)
    natr_14 = natr_14[rsi_n_day:]

    trange = talib.TRANGE(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), numpy.array(load_data_window['Close']))
    trange = trange[rsi_n_day:]

    mid_price = talib.MIDPRICE(numpy.array(load_data_window['Max']), numpy.array(load_data_window['Min']), rsi_n_day)
    mid_price = mid_price[rsi_n_day:]
    
    series_data = load_data_window[rsi_n_day:].copy()
    series = series_data.reset_index(drop=True)

    price_series = series['Close'].copy()
    date = numpy.array(series['Date'])
    price = numpy.array(series['Close'])

    # prepare data
    scaler_close, test = prepare_data(series, n_lag, n_seq, rsi_14, mfi_14, mom_14, plus_dm_14, dx_14, cci_14, aaron_14, cmo_14, roc_14, rorc_14, will_14, bop, ad, obv, avg_price, med_price, typ_price, wcl_price, atr_14, natr_14, trange, mid_price)

    forecasts = make_forecasts(model, n_batch, test, n_seq, n_lag)
    forecasts = inverse_transform(price_series, forecasts, scaler_close)

    actual = [row[-n_seq:] for row in test]
    actual = inverse_transform(price_series, actual, scaler_close)
#----------------------------------------- 
    pred_1 = []
    pred_2 = []
    pred_3 = []

    actual_1 = []
    actual_2 = []
    actual_3 = []

    difference_array1 = []
    difference_array2 = []
    difference_array3 = []

    rsi_bound_top = 85 
    rsi_bound_low = 30
    profit_boundarie = 0
    
    decision = []
    
    for i in range(len(forecasts)):
        pred_1.append(round(forecasts[i][0],1))	
        pred_2.append(round(forecasts[i][1],1))	
        pred_3.append(round(forecasts[i][2],1))		
        actual_1.append(round(actual[i][0],1))	
        actual_2.append(round(actual[i][1],1))	
        actual_3.append(round(actual[i][2],1))	
#----------------------------------------- 
    for i in range(len(forecasts)):
        difference_array1.append((pred_1[i]-actual_1[i])**2)
        difference_array2.append((pred_2[i]-actual_2[i])**2)
        difference_array3.append((pred_3[i]-actual_3[i])**2)
 
    pred_1 = pred_1[-window:]
    pred_2 = pred_2[-window:]
    pred_3 = pred_3[-window:]
    actual_1 = actual_1[-window:]
    actual_2 = actual_2[-window:]
    actual_3 = actual_3[-window:]
    rsi_14 = rsi_14[-window-n_seq:-n_seq]
    price = price[-price_window-n_seq:-n_seq]
    date = date[-window-n_seq:-n_seq]
    #window size
    for i in range(window):
        diff_value1 = abs(pred_1[i]-actual_1[i])
        diff_value2 = abs(pred_2[i]-actual_2[i])
        diff_value3 = abs(pred_3[i]-actual_3[i])
        price[i+6] = round(price[i+6],1)
        profit_actual = round(((price[i+6] - price_mem)/price_mem)*100,1)

        if  buy_allowance is True and (((rsi_14[i] < rsi_bound_low) and (price[i+6] < (pred_1[i] or pred_3[i] or pred_2[i])) or (((price[i+6]+(price[i+6]/100)*par1) < (pred_3[i] or pred_2[i] or pred_1[i])) and (price[i+6] < pred_1[i])))):	
            buy_allowance = False
            sell_allowance = True
            price_mem = price[i+6]
            decision = "Nakup "		
            trades.append(str(date[i]) + "  " + str(price[i+6]) + "  " + decision)
 	    		#and profit_actual > profit_boundarie
        elif (sell_allowance is True ) and (((price[i+6] > (pred_1[i] and pred_2[i])) and (rsi_14[i] > rsi_bound_top)) or (((price[i+6]-(price[i+6]/100)*par2) > (pred_3[i] or pred_2[i] or pred_1[i])) and ((price[i+6] > pred_1[i])or(price[i+4] > price[i+6])))):		
            buy_allowance = True
            sell_allowance = False
            profit_long += profit_actual
            decision = "Prodej   +" + str(profit_actual) + "%"
            trades.append(str(date[i]) + "  " + str(price[i+6]) + "  " + decision)	
        else:
            decision = "     "
            trades.append(str(date[i]) + "  " + str(price[i+6]) + "  " + decision)	   
        operation.append(decision)
        date_remember.append(date[i])
        price_remember.append(price[i+6])
        
        avg_diff_output_txt = open('txt/BTC_avg_diff.txt','a+')  
        avg_diff_output_txt.write(" %f ; %f ; %f \n" % (diff_value1, diff_value2, diff_value3))
        avg_diff_output_txt.close()
        
    rmse_plus_test_1 = sqrt(sum(difference_array1[-window:])/window)
    rmse_plus_test_2 = sqrt(sum(difference_array2[-window:])/window)
    rmse_plus_test_3 = sqrt(sum(difference_array3[-window:])/window)
    
    rmse_plus_train_1 = sqrt(sum(difference_array1[:-window])/len(difference_array1[:-window]))
    rmse_plus_train_2 = sqrt(sum(difference_array2[:-window])/len(difference_array1[:-window]))
    rmse_plus_train_3 = sqrt(sum(difference_array3[:-window])/len(difference_array1[:-window]))
    
    #train, test rmse to .txt file
    test_rmse_output_txt = open('txt/BTC_RMSE_output.txt','a+')  
    test_rmse_output_txt.write(" %f ; %f ; %f  ;   %f ; %f ; %f  \n" % (rmse_plus_train_1, rmse_plus_train_2, rmse_plus_train_3, rmse_plus_test_1, rmse_plus_test_2, rmse_plus_test_3))
    test_rmse_output_txt.close()
    #trades .txt
    trades = trades[-window:]
    trade_output_txt = open('txt/BTC_trade_output.txt','a+')
    trade_output_txt.write("\n")
    trade_output_txt.write("\n".join(trades))
    trade_output_txt.close()
    
    del model
    print("====== konec iterace ======")


trace1 = Scatter(x=date_remember, y=price_remember, mode='lines+text', text=operation, textposition='top center')
graf_data=go.Data([trace1])
layout=go.Layout(title= "Profit " + (str(round(profit_long,1)) +"%"), xaxis={'title':'datum'}, yaxis={'title':'Cena [EUR]'})
figure=go.Figure(data=graf_data,layout=layout)
plotly.offline.plot(figure)	