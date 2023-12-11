import streamlit as st
import httpx
import pandas as pd
import os
import requests

#UPLOAD_FOLDER = "uploads_folder"
FASTAPI_ENDPOINT = "http://localhost:8000/upload_files/"  # Replace with your FastAPI endpoint

# FastAPI server URL
fastapi_url = "http://localhost:8000"  # Replace with the actual URL of your FastAPI server

# Streamlit app
def main(upload_folder):
    st.title("FastAPI + Streamlit Integration")

    # Sidebar
    selected_page = st.sidebar.selectbox("Select Page", ["Upload", "Search"])

    if selected_page == "Upload":
        upload_page(upload_folder)
    elif selected_page == "Search":
        search_page()

def upload_page(upload_folder):
    st.title("Streamlit File Uploader and Processor")
    

    uploaded_files = st.file_uploader("Choose files to upload", type=["txt", "pdf"], accept_multiple_files=True)

    if st.button("Upload"):
        if uploaded_files:
            save_uploaded_files(uploaded_files,upload_folder)
            st.success("Files uploaded successfully!")

    st.write("### Uploaded Files:")
    #files = os.listdir(upload_folder)
    #st.write(files)

    if st.button("Process"):
        process_files(upload_folder)

def save_uploaded_files(uploaded_files,upload_folder):
    os.makedirs(upload_folder, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(upload_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

def process_files(upload_folder):
    files = os.listdir(upload_folder)
    if not files:
        st.warning("No files found in the upload folder.")
        return

    st.info(f"Processing {len(files)} files...")
    
    # Send files to FastAPI endpoint
    files_to_send = [("files", (file, open(os.path.join(upload_folder, file), "rb"))) for file in files]
    response = requests.post(FASTAPI_ENDPOINT, files=files_to_send)
    
    if response.status_code == 200:
        st.success("Files processed successfully.")
    else:
        st.error(f"Error processing files. Status code: {response.status_code}")



def search_page():
    st.header("Search Page")

    # Input for search text
    search_text = st.text_input("Enter search text:")

    if st.button("Search"):
        # Call the FastAPI endpoint for search
        response = search_in_fastapi(search_text)
        
        # Display the response as an Excel sheet
        display_search_results(response)

def search_in_fastapi(search_text):
    # FastAPI search endpoint URL
    search_endpoint = f"{fastapi_url}/search/"

    # Make a POST request to the FastAPI search endpoint
    response = httpx.post(search_endpoint, json={"text": search_text})

    # Return the JSON response
    return response.json()

def display_search_results(response):
    if response["responses"]:
        # Create a DataFrame from the search results
        df = pd.DataFrame(response["responses"])

        # Render the DataFrame as an Excel sheet using st.dataframe
        st.dataframe(df)

        # Provide a download button for the DataFrame as an Excel file
        st.download_button(
            label="Download Excel",
            data=df.to_csv(index=False).encode(),
            file_name="search_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No search results found.")

if __name__ == "__main__":
    main(upload_folder = r"C:\Users\tarun\Desktop\check\uploads_folder")