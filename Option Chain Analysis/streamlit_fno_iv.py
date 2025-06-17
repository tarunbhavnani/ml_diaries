import streamlit as st
import pandas as pd
from main import *
import os
from datetime import datetime

directory = r"C:\Users\tarun\Desktop\Option chin app"

st.set_page_config(page_title="CSV Metadata Explorer", layout="wide")
st.title("📊 CSV Metadata & Options Chain Analyzer")

# File uploader for current option chain
uploaded_file = st.file_uploader("📂 Upload CURRENT Option Chain CSV", type=["csv"])
df = None
df_prev = None
UPLOAD_FOLDER = None

if uploaded_file:
    original_name = uploaded_file.name
    UPLOAD_FOLDER = original_name.split('-')[3]
    # if not os.path.isdir(UPLOAD_FOLDER):
    #     os.mkdir(UPLOAD_FOLDER)
    try:
        df = read_nes_option_chain_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading current file: {e}")
        df = None

# Load saved prices CSV
saved_prices = pd.read_csv("saved_prices.csv")

# Optional: File uploader for previous option chain
previous_uploaded_file = st.file_uploader("📂 (Optional) Upload PREVIOUS Option Chain CSV", type=["csv"])
if previous_uploaded_file is not None:
    try:
        df_prev = read_nes_option_chain_csv(previous_uploaded_file)
    except Exception as e:
        st.warning(f"⚠️ Could not read previous file: {e}")
        df_prev = None

# If current file is uploaded and parsed
if df is not None:
    default_spot = None
    spot_price = st.number_input(
        "🎯 Enter Spot Price (required to proceed)",
        value=default_spot,
        step=1,
        format="%d"
    )

    # ✅ Execute button to trigger backend logic
    if spot_price and st.button("🚀 Execute Analysis"):
        try:
            
            saved_prices = add_or_update(saved_prices, uploaded_file.name, spot_price)
            
            saved_prices.to_csv('saved_prices.csv', index=False)
            
            #saved_prices.loc[len(saved_prices)] = [uploaded_file.name, spot_price]
            #saved_prices.to_csv('saved_prices.csv', index=False)

            df = get_relevant_strikes(df, spot_price, percent_range=10)
            df_call, df_put = call_put_demerge(df)

            

            df_up = re_df(df_call, df_put)
            st.subheader("📊 Option Chain")
            st.dataframe(df_up, use_container_width=True)

            # if not os.path.isdir(UPLOAD_FOLDER):
            #     os.mkdir(UPLOAD_FOLDER)
            # filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_reformatted.csv"
            # save_path = os.path.join(UPLOAD_FOLDER, filename)
            # df_up.to_csv(save_path, index=False)
            # st.success(f"✅ Reformatted Option Chain saved to `{save_path}`")

            
            if df_prev is None:
                df_prev = no_df_prev(df.copy())
                spot_price_prev = spot_price  # fallback if no prior data available
            else:
                # Try to get the previous spot price from saved_prices
                match = saved_prices[saved_prices['file'] == previous_uploaded_file.name]
                if not match.empty:
                    spot_price_prev = match['price'].iloc[0]
                else:
                    spot_price_prev = spot_price  # fallback if no previous price found


                summary=quadrant_oi_price_change(df, df_prev, spot_price=spot_price, spot_prev=spot_price_prev, percent_range=10, ntm_range=2)
                
                st.subheader("📈 Delta Summary")
                st.json(summary)
            
            # Run the analysis using the previous and current data
            analysis = compare_previous_oi(df, df_prev, spot_price, spot_price_prev)            
            st.subheader("📉 Delta Analysis")
            st.dataframe(analysis, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing analysis: {e}")








