# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:23:07 2025

@author: tarun
"""

import pandas as pd
import numpy as np
from datetime import datetime

import os
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

def compare_previous_oi(df, df_prev, spot_price):
    
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
                   
     
    return analysis
    

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

def quadrant_oi_price(df, spot_price, percent_range=5):
    df = get_relevant_strikes(df, spot_price, percent_range)
    df_call, df_put = call_put_demerge(df)

    def format_pct(value):
        return f"{round(value, 2)}%"

    summ = {}
    
    # ITM Put
    data = df_put[df_put['STRIKE'] >= spot_price]
    oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
    price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
    summ['itm_put_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }

    # OTM Put
    data = df_put[df_put['STRIKE'] < spot_price]
    oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
    price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
    summ['otm_put_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }

    # ITM Call
    data = df_call[df_call['STRIKE'] < spot_price]
    oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
    price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
    summ['itm_call_delta_%'] = {
        "OI": format_pct(oi_pct),
        "Price": format_pct(price_pct)
    }

    # OTM Call
    data = df_call[df_call['STRIKE'] >= spot_price]
    oi_pct = 100 * sum(data['CHNG IN OI']) / sum(data['OI'])
    price_pct = 100 * np.mean(data['CHNG'] / (data['LTP'] - data['CHNG']))
    summ['otm_call_delta_%'] = {
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


# path=r"C:\Users\tarun\Desktop\Option chin app\option-chain-ED-NIFTY-22-May-2025 (1).csv"
# spot_price= 25000

# path=r"C:\Users\tarun\Desktop\Option chin app\option-chain-ED-BAJFINANCE-29-May-2025.csv"
# spot_price=8641


# df= read_nes_option_chain_csv(path)

# df= get_relevant_strikes(df, spot_price, percent_range=10)

# df_call, df_put=call_put_demerge(df)


# summary= quadrant_oi_price(df, spot_price)

# summary['PCR_OI'], summary['PCR_Volume']=PCR(df, spot_price, percent_range=5)



# summary['max_pain_strike'], summary['max_pain_strike_call'], summary['max_pain_strike_put'] =max_pain(df_call, df_put, spot_price)


# summary['iv_atm_ce'], summary['iv_atm_pe']= iv_skew(df_call, df_put, spot_price)
