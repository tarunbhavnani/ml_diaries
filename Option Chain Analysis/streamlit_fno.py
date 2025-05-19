import streamlit as st
import pandas as pd
from main import *

st.set_page_config(page_title="CSV Metadata Explorer", layout="wide")
st.title("📊 CSV Metadata & Options Chain Analyzer")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV using your custom parser
        df = read_nes_option_chain_csv(uploaded_file)

        # Show initial preview
        st.subheader("📋 File Preview")
        st.dataframe(df.head(5), use_container_width=True)

        # Suggest default spot price based on median STRIKE
        default_spot = int(df['STRIKE'].median())

        # Spot price input
        spot_price = st.number_input(
            "🎯 Enter Spot Price (required to proceed)",
            value=default_spot,
            step=1,
            format="%d"
        )

        if spot_price:
            # Proceed with processing
            df = get_relevant_strikes(df, spot_price, percent_range=10)
            df_call, df_put = call_put_demerge(df)
            
            

            summary= quadrant_oi_price(df, spot_price)
            summary["PCR_OI"], summary["PCR_Volume"] = PCR(df, spot_price, percent_range=5)
            summary["max_pain_strike"], summary["max_pain_strike_call"], summary["max_pain_strike_put"] = max_pain(df_call, df_put, spot_price)
            summary["iv_atm_ce"], summary["iv_atm_pe"] = iv_skew(df_call, df_put, spot_price)

            # Metadata
            st.subheader("🧾 CSV Metadata")
            st.markdown(f"**Shape:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")
            st.markdown("**Columns:** " + ", ".join(f"`{col}`" for col in df.columns))

            # Sample Data
            st.subheader("📊 Full Option Chain (filtered)")
            st.dataframe(df, use_container_width=True)

            # Summary
            st.subheader("📈 Analysis Summary")
            st.json(summary)
            
            df_up=re_df(df_call, df_put)
            # updates Data
            st.subheader("📊 Full Option Chain (filtered)")
            st.dataframe(df_up, use_container_width=True)
            
            

    except Exception as e:
        st.error(f"❌ Error reading or processing the CSV: {e}")