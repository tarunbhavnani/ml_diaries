import streamlit as st
import pandas as pd
from main import *
import os
from datetime import datetime

st.set_page_config(page_title="CSV Metadata Explorer", layout="wide")
st.title("📊 CSV Metadata & Options Chain Analyzer")

# File uploader for current option chain
uploaded_file = st.file_uploader("📂 Upload CURRENT Option Chain CSV", type=["csv"])
if uploaded_file:
    original_name = uploaded_file.name
    UPLOAD_FOLDER=original_name.split('-')[3]
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)


# Optional: File uploader for previous option chain
previous_uploaded_file = st.file_uploader("📂 (Optional) Upload PREVIOUS Option Chain CSV", type=["csv"])
df_prev = None

if previous_uploaded_file is not None:
    try:
        df_prev = read_nes_option_chain_csv(previous_uploaded_file)
    except Exception as e:
        st.warning(f"⚠️ Could not read previous file: {e}")

if uploaded_file is not None:
    try:
        df = read_nes_option_chain_csv(uploaded_file)

        #st.subheader("📋 File Preview")
        #st.dataframe(df.head(5), use_container_width=True)

        default_spot = None

        spot_price = st.number_input(
            "🎯 Enter Spot Price (required to proceed)",
            value=default_spot,
            step=1,
            format="%d"
        )

        if spot_price:
            df = get_relevant_strikes(df, spot_price, percent_range=10)
            df_call, df_put = call_put_demerge(df)

            summary = quadrant_oi_price(df, spot_price)
            summary["PCR_OI"], summary["PCR_Volume"] = PCR(df, spot_price, percent_range=5)
            summary["max_pain_strike"], summary["max_pain_strike_call"], summary["max_pain_strike_put"] = max_pain(df_call, df_put, spot_price)
            summary["iv_atm_ce"], summary["iv_atm_pe"] = iv_skew(df_call, df_put, spot_price)

            st.subheader("🧾 CSV Metadata")
            st.markdown(f"**Shape:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")
            st.markdown("**Columns:** " + ", ".join(f"`{col}`" for col in df.columns))

            #st.subheader("📊 Full Option Chain (filtered)")
            #st.dataframe(df, use_container_width=True)

            st.subheader("📈 Analysis Summary")
            st.json(summary)

            df_up = re_df(df_call, df_put)
            st.subheader("📊 Reformatted Option Chain")
            st.dataframe(df_up, use_container_width=True)
            #savefile(df_up, UPLOAD_FOLDER,file_format="csv")
            # ✅ Save df_up to folder after creating it
            if not os.path.isdir(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_reformatted.csv"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            df_up.to_csv(save_path, index=False)
            st.success(f"✅ Reformatted Option Chain saved to `{save_path}`")

            if df_prev is not None:
                analysis = compare_previous_oi(df, df_prev, spot_price)
                st.subheader("📉 IV/OI Comparative Analysis")
                st.dataframe(analysis, use_container_width=True)
                #savefile(analysis, UPLOAD_FOLDER,file_format="csv")

    except Exception as e:
        st.error(f"❌ Error reading or processing the CSV: {e}")
