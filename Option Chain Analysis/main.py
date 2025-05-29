# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:23:07 2025

@author: tarun
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime

import os
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np


# =============================================================================
# # Function to add or update the file
# =============================================================================
def add_or_update(df, file, price):
    dic = {i: j for i, j in zip(df.file, df.price)}
    dic[file] = price  # Corrected this line
    df = pd.DataFrame(dic.items(), columns=['file', 'price'])
    return df

# =============================================================================
# get price latest
# =============================================================================
def get_price(url):
    response=requests.get(url)
    
    soup=BeautifulSoup(response.text, 'html.parser')
    class1="YMlKec fxKbKc"
    
    price=soup.find(class_=class1).text
    
    price= float(price[1:])
    
    return price
    

# =============================================================================
# Blacj scholes option pricing 
# =============================================================================

# Black-Scholes formula
def bs_call_price(S, K, T, sigma,r=0):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Find IV given option price
def implied_volatility(target_price, S, K, T, r=0):
    f = lambda sigma: bs_call_price(S, K, T, r, sigma) - target_price
    return brentq(f, 0.0001, 3)


# =============================================================================
# read csv
# =============================================================================

def read_nes_option_chain_csv(path):
    df = pd.read_csv(path, skiprows=1)
    df = df.replace(',', '', regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    return df
    

# =============================================================================
# code to pull out all the strikes in 10 pc range of the spot
# =============================================================================



def get_relevant_strikes(df, spot_price, percent_range=10):
    """
    Returns a list of strike prices within a specified percentage range of the spot price.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'STRIKE' column.
        spot_price (float): Current spot price.
        percent_range (float): Percentage range above and below the spot price to filter strikes.

    Returns:
        List[float]: Filtered list of relevant strike prices.
    """
    upper_bound = spot_price * (1 + percent_range / 100)
    lower_bound = spot_price * (1 - percent_range / 100)
    strikes = df['STRIKE'].unique()
    relevant_strikes = [strike for strike in strikes if lower_bound <= strike <= upper_bound]
    
    # if len(strikes)>40:
    #     center_index = relevant_strikes.index(spot_price)
    
    #     # Slice the list to get 20 before and 20 after
    #     relevant_strikes = relevant_strikes[center_index - 25 : center_index + 25]
    
    
    return df[df['STRIKE'].isin(relevant_strikes)]
    #return sorted(relevant_strikes)




# =============================================================================
# us OI threshold of 50 percentile as well
# =============================================================================

def get_relevant_data(df, spot_price, percent_range=10,threshold_percentile=50):
    oi= df['OI']+df['OI.1']
    thresh=np.percentile(oi, threshold_percentile)
    

    upper_bound = spot_price * (1 + percent_range / 100)
    lower_bound = spot_price * (1 - percent_range / 100)
    strikes = df['STRIKE'].unique()
    relevant_strikes = [strike for strike in strikes if lower_bound <= strike <= upper_bound]
    
    df= df[oi>thresh]
    
    
    return df[df['STRIKE'].isin(relevant_strikes)]
    #return sorted(relevant_strikes)




# =============================================================================
# IV change analysis
# =============================================================================
#sentiment for comapring
# Function to assign sentiment score
# def get_sentiment_price_oi(row):
#     # Extract values
#     call_ltp = row["ltp_change_call"]
#     call_oi = row["oi_change_call"]
#     put_ltp = row["ltp_change_put"]
#     put_oi = row["oi_change_put"]

#     # Signal mapping
#     def interpret_signal(ltp, oi):
#         if ltp > 0 and oi > 0:
#             return 2  # Long build-up
#         elif ltp > 0 and oi < 0:
#             return 2  # Short covering
#         elif ltp < 0 and oi > 0:
#             return -2  # Short build-up
#         elif ltp < 0 and oi < 0:
#             return -2  # Long unwinding
#         else:
#             return 0  # Neutral / unclear

#     call_score = interpret_signal(call_ltp, call_oi)
#     put_score = -interpret_signal(put_ltp, put_oi)  # reverse logic for puts

#     total_score = call_score + put_score

#     # Final sentiment verdict
#     if total_score >= 4:
#         return "🔥 Strong Bullish"
#     elif total_score > 0:
#         return "🟢 Bullish"
#     elif total_score == 0:
#         return "⚪ Neutral"
#     elif total_score < -3:
#         return "🔻 Strong Bearish"
#     else:
#         return "🔻 Bearish"


# def get_sentiment_price_oi(row):
#     # Extract values
#     call_ltp = row["ltp_change_call"]
#     call_oi = row["oi_change_call"]
#     put_ltp = row["ltp_change_put"]
#     put_oi = row["oi_change_put"]

#     #call side
#     call_analysis=""
#     if call_ltp>0 and call_oi >0:
#         call_analysis= "Bullish,call buying"

#     if call_ltp>0 and call_oi <0:
#         call_analysis= "Neutral,Profit Booking"
    
#     if call_ltp<0 and call_oi >0:
#         call_analysis= "Bearish,call writing"
    
#     if call_ltp<0 and call_oi <0:
#         call_analysis= "Unclear,call unwinding"
    
#     #put side
#     put_analysis=""
    
#     if put_ltp>0 and put_oi>0:
#         put_analysis= "Bearish,put buying"

#     if put_ltp>0 and put_oi<0:
#         put_analysis= "Neutral,profit booking"

#     if put_ltp<0 and put_oi>0:
#         put_analysis= "Bullish,put writing"

#     if put_ltp<0 and put_oi<0:
#         put_analysis= "Unclear,put unwinding"
    
    
#     return call_analysis, put_analysis

# def analyze_call_side(call_ltp, call_oi, call_iv):
#     if call_ltp > 0:
#         if call_oi > 0:
#             return "🔥 Bullish: Call Buying with rising IV" if call_iv > 1 else "⚪ Bullish: Call Buying with steady IV"
#         elif call_oi < 0:
#             return "⚪ Neutral: Profit Booking in Calls" if call_iv > 1 else "🟡 Neutral: Call Unwinding"
#     elif call_ltp < 0:
#         if call_oi > 0:
#             return "⚪ Bearish: Call Writing (rising IV)" if call_iv > 1 else "🔴 Bearish: Call Writing (aggressive)"
#         elif call_oi < 0:
#             return "📉 Bearish: Call Unwinding"
#     return "⏸️ Neutral or Inconclusive"

# def analyze_put_side(put_ltp, put_oi, put_iv):
#     if put_ltp > 0:
#         if put_oi > 0:
#             return "🔴 Bearish: Put Buying with rising IV" if put_iv > 1 else "⚪ Bearish: Put Buying with steady IV"
#         elif put_oi < 0:
#             return "⚪ Neutral: Profit Booking in Puts" if put_iv > -1 else "🟡 Neutral: Put Unwinding"
#     elif put_ltp < 0:
#         if put_oi > 0:
#             return "⚪ Bullish: Put Writing (rising IV)" if put_iv > 1 else "🔥 Bullish: Put Writing (aggressive)"
#         elif put_oi < 0:
#             return "📈 Bullish: Put Unwinding"
#     return "⏸️ Neutral or Inconclusive"

def analyze_call_side(call_ltp, call_oi, call_iv):
    IV_RISE = 5
    IV_FALL = -5

    if call_ltp > 0:
        if call_oi > 0:
            if call_iv > IV_RISE:
                return "🔥 Bullish: Call Buying with Rising IV"
            elif call_iv < IV_FALL:
                return "🟢 Bullish: Demand-driven (IV falling)"
            else:
                return "⚪ Bullish: Call Buying with Steady IV"
        elif call_oi < 0:
            if call_iv > IV_RISE:
                return "🟠 Neutral: Profit Booking in Calls (IV rising)"
            elif call_iv < IV_FALL:
                return "🟡 Neutral: Call Unwinding with IV Drop"
            else:
                return "⚪ Neutral: Call Unwinding"
    elif call_ltp < 0:
        if call_oi > 0:
            if call_iv > IV_RISE:
                return "🔴 Bearish: Call Writing with IV Spike"
            elif call_iv < IV_FALL:
                return "🟡 Neutral: IV-led Price Drop, Mild Writing"
            else:
                return "⚪ Bearish: Call Writing"
        elif call_oi < 0:
            if call_iv < IV_FALL:
                return "📉 Bearish: Call Unwinding with IV Crush"
            else:
                return "📉 Bearish: Call Unwinding"
    return "⏸️ Neutral or Inconclusive"

def analyze_put_side(put_ltp, put_oi, put_iv):
    IV_RISE = 2
    IV_FALL = -2

    if put_ltp > 0:
        if put_oi > 0:
            if put_iv > IV_RISE:
                return "🔴 Bearish: Put Buying with IV Spike"
            elif put_iv < IV_FALL:
                return "🟠 Bearish: Demand-led Put Buying (IV down)"
            else:
                return "⚪ Bearish: Put Buying with Steady IV"
        elif put_oi < 0:
            if put_iv > IV_RISE:
                return "🟡 Neutral: Profit Booking in Puts (IV up)"
            elif put_iv < IV_FALL:
                return "🟢 Neutral: Put Unwinding with IV Drop"
            else:
                return "⚪ Neutral: Put Unwinding"
    elif put_ltp < 0:
        if put_oi > 0:
            if put_iv > IV_RISE:
                return "🔥 Bullish: Put Writing with IV Spike"
            elif put_iv < IV_FALL:
                return "🟢 Bullish: IV-led Drop, Strong Put Writing"
            else:
                return "⚪ Bullish: Put Writing"
        elif put_oi < 0:
            if put_iv < IV_FALL:
                return "📈 Mildly Bullish: Put Unwinding with IV Crush"
            else:
                return "⚪ Neutral: Put Unwinding (OI ↓, LTP ↓)"
    return "⏸️ Neutral or Inconclusive"
def get_sentiment_price_oi(row):
    # Extract values
    call_ltp = row["ltp_change_call"]
    call_oi = row["oi_change_call"]
    call_iv = row["iv_change_call"]
    
    put_ltp = row["ltp_change_put"]
    put_oi = row["oi_change_put"]
    put_iv= row["iv_change_put"]    
    
    call_analysis=analyze_call_side(call_ltp, call_oi, call_iv)
    put_analysis=analyze_put_side(put_ltp, put_oi, put_iv)
    
    return call_analysis, put_analysis


def compare_previous_oi(df, df_prev, spot_price, spot_price_prev=None):
    if spot_price_prev:
        price_update= spot_price- spot_price_prev
    else:
        price_update= 0
    
    df=get_relevant_strikes(df, spot_price, percent_range=10)
    df_prev=get_relevant_strikes(df_prev, spot_price, percent_range=10)
    
    list(df)
    df_call, df_put=call_put_demerge(df)
    
    df_call_prev, df_put_prev=call_put_demerge(df_prev)
    
    #call analysis
    
    oi_change= df_call['OI']-df_call_prev['OI']
    ltp_change= df_call['LTP']-df_call_prev['LTP']
    iv_change= df_call['IV']-df_call_prev['IV']
    
    call_analysis=pd.DataFrame({'STRIKE':df_call.STRIKE, 'ltp_change_call':ltp_change, 'oi_change_call':oi_change, 'iv_change_call':iv_change})
    
    
    #put analysis
    
    oi_change= df_put['OI']-df_put_prev['OI']
    ltp_change= df_put['LTP']-df_put_prev['LTP']
    iv_change= df_put['IV']-df_put_prev['IV']
    
    put_analysis=pd.DataFrame({'STRIKE':df_put.STRIKE, 'ltp_change_put':ltp_change, 'oi_change_put':oi_change, 'iv_change_put':iv_change})
    
    analysis= call_analysis.merge(put_analysis, on='STRIKE')
    analysis=analysis[['oi_change_call','iv_change_call','ltp_change_call','STRIKE','ltp_change_put','iv_change_put','oi_change_put']]
                   
    sentiment=analysis.apply(get_sentiment_price_oi, axis=1)
    analysis["Call-Sentiment"]= [i[0] for i in sentiment]
    analysis["Put-Sentiment"]=  [i[1] for i in sentiment]
    
    analysis['price_update']=price_update
    analysis=analysis[['Call-Sentiment', 'oi_change_call','iv_change_call','ltp_change_call','price_update','STRIKE','ltp_change_put','iv_change_put','oi_change_put','Put-Sentiment']]
    return analysis
    

def no_df_prev(df):
    df['OI']=df['OI']-df['CHNG IN OI']
    df['OI.1']=df['OI.1']-df['CHNG IN OI.1']
    df['LTP']=df['LTP']-df['CHNG']
    df['LTP.1']=df['LTP.1']-df['CHNG.1']
    return df


# =============================================================================
# demerge call put data
# =============================================================================

def call_put_demerge(df):
    
    df_call= df[['OI','CHNG IN OI','VOLUME','IV','LTP','CHNG','BID QTY','BID','ASK','ASK QTY', 'STRIKE']]

    df_put= df[['BID QTY.1','BID.1','ASK.1','ASK QTY.1','CHNG.1','LTP.1','IV.1','VOLUME.1','CHNG IN OI.1','OI.1']]

    df_put.columns= [i.split('.')[0] for i in df_put]
    
    df_put['STRIKE']= df.STRIKE
    
    return df_call, df_put
    


# =============================================================================
# max pain
# =============================================================================

def max_pain(df_call, df_put, spot_price):
    
    combined = pd.merge(df_call[['STRIKE', 'OI']], df_put[['STRIKE', 'OI']], on='STRIKE', suffixes=('_CE', '_PE'))
    combined['Total_OI'] = combined['OI_CE'] + combined['OI_PE']
    max_pain_strike = combined.loc[combined['Total_OI'].idxmax(), 'STRIKE']
    max_pain_strike_call=df_call[df_call['OI']==max(df_call['OI'])]['STRIKE'].iloc[0]
    max_pain_strike_put=df_put[df_put['OI']==max(df_put['OI'])]['STRIKE'].iloc[0]
    
    return max_pain_strike, max_pain_strike_call, max_pain_strike_put
    



# =============================================================================
#  # 2. Put Call Ratio (PCR)
# =============================================================================


def PCR(df, spot_price, percent_range=5):
    df= get_relevant_strikes(df, spot_price, percent_range)
    df_call, df_put=call_put_demerge(df)
    total_put_oi = df_put['OI'].sum()
    total_call_oi = df_call['OI'].sum()
    total_put_vol = df_put['VOLUME'].sum()
    total_call_vol = df_call['VOLUME'].sum()
    
    PCR_OI = round(total_put_oi / total_call_oi, 2)
    PCR_Volume = round(total_put_vol / total_call_vol, 2)
    
    return PCR_OI, PCR_Volume



# =============================================================================
# ITM puts
# =============================================================================
import numpy as np
# df_put_itm=df_put[df_put['STRIKE']<spot_price]

# df_put_itm['OI_pc_delta']= df_put_itm['CHNG IN OI']/df_put_itm['OI']
# #df_put_itm['price_pc_delta']= df_put_itm['CHNG']/(df_put_itm['LTP']-df_put_itm['CHNG'])

# def quadrant_oi_price(df, spot_price,percent_range=5):
#     df= get_relevant_strikes(df, spot_price, percent_range)
#     df_call, df_put=call_put_demerge(df)

#     summ={}
#     #itm_put
#     data=df_put[df_put['STRIKE']>=spot_price]
#     summ['itm_put_delta_%']={"OI":100*sum(data['CHNG IN OI'])/sum(data['OI']), "Price":100*np.mean(data['CHNG']/(data['LTP']-data['CHNG']))}
    
#     #otm_put
#     data=df_put[df_put['STRIKE']<spot_price]
#     summ['otm_put_delta_%']={"OI":100*sum(data['CHNG IN OI'])/sum(data['OI']), "Price":100*np.mean(data['CHNG']/(data['LTP']-data['CHNG']))}

#     #itm_call
#     data=df_call[df_call['STRIKE']<spot_price]
#     summ['itm_call_delta_%']={"OI":100*sum(data['CHNG IN OI'])/sum(data['OI']), "Price":100*np.mean(data['CHNG']/(data['LTP']-data['CHNG']))}

#     #otm_call
#     data=df_call[df_call['STRIKE']>=spot_price]
#     summ['otm_call_delta_%']={"OI":100*sum(data['CHNG IN OI'])/sum(data['OI']), "Price":100*np.mean(data['CHNG']/(data['LTP']-data['CHNG']))}
    
#     return summ

# def quadrant_oi_price(df, spot_price, percent_range=5):
#     df = get_relevant_strikes(df, spot_price, percent_range)
#     df_call, df_put = call_put_demerge(df)

#     def format_pct(value):
#         return f"{round(value, 2)}%"

#     summ = {}
    
#     # ITM Put
#     data = df_put[df_put['STRIKE'] >= spot_price]
#     oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
#     price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
#     summ['itm_put_delta_%'] = {
#         "OI": format_pct(oi_pct),
#         "Price": format_pct(price_pct)
#     }

#     # OTM Put
#     data = df_put[df_put['STRIKE'] < spot_price]
#     oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
#     price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
#     summ['otm_put_delta_%'] = {
#         "OI": format_pct(oi_pct),
#         "Price": format_pct(price_pct)
#     }

#     # ITM Call
#     data = df_call[df_call['STRIKE'] < spot_price]
#     oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
#     price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
#     summ['itm_call_delta_%'] = {
#         "OI": format_pct(oi_pct),
#         "Price": format_pct(price_pct)
#     }

#     # OTM Call
#     data = df_call[df_call['STRIKE'] >= spot_price]
#     oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
#     price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
#     summ['otm_call_delta_%'] = {
#         "OI": format_pct(oi_pct),
#         "Price": format_pct(price_pct)
#     }

#     return summ


def quadrant_oi_price(df, spot_price, percent_range=10, ntm_range=2):
    df = get_relevant_strikes(df, spot_price, percent_range)
    df_call, df_put = call_put_demerge(df)

    def format_pct(value):
        return f"{round(value, 2)}%"
    
    closest_strike= [(i,abs(spot_price-i)) for i in df.STRIKE]
    closest_strike=sorted(closest_strike, key= lambda x: x[1])[0][0]
    
    strike_diff= sorted([i for i in df.STRIKE])[0:2]
    strike_diff=abs(strike_diff[0]-strike_diff[1])
    
    
    #ntm call
    
    ntm_call_strikes= [closest_strike+i*strike_diff for i in range(ntm_range+1)]
    ntm_put_strikes= [closest_strike-i*strike_diff for i in range(ntm_range+1)]
    
    
    ntm_call_df= df_call[df_call.STRIKE.isin(ntm_call_strikes)]
    ntm_put_df= df_put[df_put.STRIKE.isin(ntm_put_strikes)]
    
    
       
    
    summ = {}
    
    # ntm call
    
    oi_pct = 100 * sum(ntm_call_df['CHNG IN OI']) / sum(ntm_call_df['OI'])
    price_pct = 100 * np.mean(ntm_call_df['CHNG'] / (ntm_call_df['LTP'] - ntm_call_df['CHNG']))
    summ['ntm_call_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }

    # ntm Put
    
    oi_pct = 100 * sum(ntm_put_df['CHNG IN OI']) / sum(ntm_put_df['OI'])
    price_pct = 100 * np.mean(ntm_put_df['CHNG'] / (ntm_put_df['LTP'] - ntm_put_df['CHNG']))
    summ['ntm_put_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }



    otm_call_df= df_call[~df_call.STRIKE.isin(ntm_call_strikes)]
    otm_call_df=otm_call_df[otm_call_df.STRIKE<min(ntm_call_strikes)]
    
    otm_put_df= df_put[~df_put.STRIKE.isin(ntm_put_strikes)]
    otm_put_df=otm_put_df[otm_put_df.STRIKE>max(ntm_put_strikes)]


    # otm call
    
    oi_pct = 100 * sum(otm_call_df['CHNG IN OI']) / sum(otm_call_df['OI'])
    price_pct = 100 * np.mean(otm_call_df['CHNG'] / (otm_call_df['LTP'] - otm_call_df['CHNG']))
    summ['otm_call_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }

    # otm Put
    
    oi_pct = 100 * sum(otm_put_df['CHNG IN OI']) / sum(otm_put_df['OI'])
    price_pct = 100 * np.mean(otm_put_df['CHNG'] / (otm_put_df['LTP'] - otm_put_df['CHNG']))
    summ['otm_put_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }



    return summ
# =============================================================================
# iv skew
# =============================================================================

def iv_skew(df_call, df_put, spot_price):
    atm_strike = min(df_call['STRIKE'], key=lambda x: abs(x - spot_price))
    iv_atm_ce = df_call.loc[df_call['STRIKE'] == atm_strike, 'IV'].values[0]
    iv_atm_pe = df_put.loc[df_put['STRIKE'] == atm_strike, 'IV'].values[0]
    
    return iv_atm_ce, iv_atm_pe


# =============================================================================
# re create df
# =============================================================================

def re_df(df_call, df_put):
    df_call.columns= [i+'-call' for i in df_call]
    #df_call.rename(columns={'STRIKE-call': 'STRIKE'}, inplace=True)
    df_call = df_call.rename(columns={'STRIKE-call': 'STRIKE'})
    df_put.columns= [i+'-put' for i in df_put]
    #df_put.rename(columns={'STRIKE-put': 'STRIKE'}, inplace=True)
    df_put = df_put.rename(columns={'STRIKE-put': 'STRIKE'})
    re_df= df_call.merge(df_put, on='STRIKE') 
    
    return re_df


# =============================================================================
# save files
# =============================================================================

def savefile(df, UPLOAD_FOLDER,file_format="csv"):
    timestamp = datetime.now().strftime("%H%M")
    filename = f"{UPLOAD_FOLDER}_{timestamp}.{file_format}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df.to_csv(filepath, index=False)
    return filepath



# =============================================================================
# final calculations
# =============================================================================

# import pandas as pd


# path=r"C:\Users\tarun\Desktop\Option chin app\option-chain-ED-JIOFIN-29-May-2025 (34).csv"
# spot_price= 293

# path_prev= r"C:\Users\tarun\Desktop\Option chin app\option-chain-ED-JIOFIN-29-May-2025 (33).csv"
# spot_price_prev= 285

# df= read_nes_option_chain_csv(path)
# df_prev= read_nes_option_chain_csv(path_prev)
# df_prev=no_df_prev(df.copy())

# df= get_relevant_strikes(df, spot_price, percent_range=10)

# df_call, df_put=call_put_demerge(df)


# delta=compare_previous_oi(df, df_prev, spot_price)


# summary= quadrant_oi_price(df, spot_price)

# summary['PCR_OI'], summary['PCR_Volume']=PCR(df, spot_price, percent_range=5)



# summary['max_pain_strike'], summary['max_pain_strike_call'], summary['max_pain_strike_put'] =max_pain(df_call, df_put, spot_price)


# summary['iv_atm_ce'], summary['iv_atm_pe']= iv_skew(df_call, df_put, spot_price)
