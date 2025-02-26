# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:30:46 2024

@author: tarun
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import base64
import time

# Function to simulate data processing
def process_data(c1c_file, optima_file, month1, month2, cag_id):
    # Simulate data processing (replace with actual processing logic)
    time.sleep(3)  # Simulate processing time
    result_data = pd.DataFrame({'CAG ID': [cag_id], 'Result': ['Some Result']})
    return result_data

# Streamlit UI
def main():
    # Page title
    st.title('Data Processing App')

    # File uploads
    st.sidebar.header('Upload Files')
    c1c_file = st.sidebar.file_uploader('Upload C1C file', type=['csv', 'xlsx'])
    optima_file = st.sidebar.file_uploader('Upload Optima file', type=['csv', 'xlsx'])

    # Month selection
    st.sidebar.header('Select Months')
    month1 = st.sidebar.selectbox('Select Month 1', ['January', 'February', 'March'])  # Add more months as needed
    month2 = st.sidebar.selectbox('Select Month 2', ['January', 'February', 'March'])  # Add more months as needed

    # CAG ID input
    st.sidebar.header('Enter CAG ID')
    cag_id = st.sidebar.text_input('CAG ID')

    # Process button
    if st.sidebar.button('Process Data'):
        if c1c_file is not None and optima_file is not None and cag_id:
            # Display loading spinner
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)  # Simulate processing time
                progress_bar.progress(percent_complete + 1)

            # Processing data
            result_data = process_data(c1c_file, optima_file, month1, month2, cag_id)

            # Download link for the result
            st.subheader('Result')
            st.write(result_data)

            # Download link
            csv = result_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download Result</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error('Please upload both C1C and Optima files and enter a CAG ID.')

if __name__ == '__main__':
    main()