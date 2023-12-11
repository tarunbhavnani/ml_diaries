# import streamlit as st
# import os
# from typing import List



# def upload_files(upload_folder: str):
#     st.title("File Uploader")

#     uploaded_files = st.file_uploader("Choose files to upload", type=["txt", "pdf"], accept_multiple_files=True)
#     for file in uploaded_files:
#         file_path = os.path.join(upload_folder, file.name)
#         with open(file_path, "wb") as f:
#             f.write(file.getvalue())
#     display_file_details(upload_folder)

# def display_file_details(upload_folder: str):
#     st.header("Uploaded Files Details")

#     files = os.listdir(upload_folder)
#     for file in files:
#         file_path = os.path.join(upload_folder, file)
#         file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
#         st.write(f"File: {file}, Size: {file_size:.2f} MB")

# def main():
#     upload_folder = r"C:\Users\tarun\Desktop\check\uploads_folder"
#     upload_files(upload_folder)

# if __name__ == "__main__":
#     main()


import streamlit as st
import os
import requests

#UPLOAD_FOLDER = "uploads_folder"
FASTAPI_ENDPOINT = "http://localhost:8000/upload_files/"  # Replace with your FastAPI endpoint

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

def main(upload_folder):
    st.title("Streamlit File Uploader and Processor")
    

    uploaded_files = st.file_uploader("Choose files to upload", type=["txt", "pdf"], accept_multiple_files=True)

    if st.button("Upload"):
        if uploaded_files:
            save_uploaded_files(uploaded_files,upload_folder)
            st.success("Files uploaded successfully!")

    st.write("### Uploaded Files:")
    files = os.listdir(upload_folder)
    st.write(files)

    if st.button("Process"):
        process_files(upload_folder)

if __name__ == "__main__":
    main(upload_folder = r"C:\Users\tarun\Desktop\check\uploads_folder")