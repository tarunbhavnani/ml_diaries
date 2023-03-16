
#filetb
Class Filetb
Description
The Filetb class is a utility class that provides functions for processing and searching text data. It contains methods for cleaning and processing text data from files, stemming and splitting sentences, and calculating cosine similarity scores for search queries.

Class Methods
__init__()
The constructor method initializes the class variables files, tb_index, all_sents, vec, tfidf_matrix, and stopwords to None and ['i', 'me', 'my', 'myself'], respectively.

stem(sent)
This method accepts a sentence as input and returns the stemmed version of the sentence. It first removes all non-alphanumeric characters and converts the sentence to lowercase. Then it calls the parsing.stem_text() function to stem the words in the sentence.

clean(sent)
This method accepts a sentence as input and returns the cleaned version of the sentence. It removes HTML tags and URLs, newlines, content inside parentheses and brackets, and non-alphanumeric characters. Finally, it removes multiple periods and returns the cleaned sentence.

split_into_sentences(text)
This method accepts a string of text as input and returns a list of sentences extracted from the text. It splits the text into sentences using the period and space character as a delimiter, removes any sentence shorter than 20 characters, and removes any numbering within the sentence.

files_processor_tb(files)
This method accepts a list of file paths as input and processes the text data from the files. It iterates through the files, reads each page of the file, cleans and splits the text into sentences, and creates an index of the sentences along with the file name and page number. It also creates a term frequency-inverse document frequency (TF-IDF) vectorizer and transforms the sentences into a TF-IDF matrix. It then assigns the processed data and vectorizer to the class variables tb_index, all_sents, vec, and tfidf_matrix, respectively.

get_response_cosine(question, min_length=7, score_threshold=0.1)
This method accepts a question as input and returns a list of sentences that are most relevant to the question based on cosine similarity scores. It first cleans and stems the question, then calculates the cosine similarity scores between the question vector and the sentence vectors in the TF-IDF matrix. It then returns a list of relevant sentences whose length is greater than or equal to min_length and whose cosine similarity score is greater than score_threshold.

Class Variables
files
This variable stores the list of file paths passed to the files_processor_tb method.

tb_index
This variable stores the index of the processed sentences in the format of a list of dictionaries with keys doc, page, and sentence.

all_sents
This variable stores the list of all processed sentences.

vec
This variable stores the TF-IDF vectorizer.

tfidf_matrix
This variable stores the TF-IDF matrix.

stopwords
This variable stores the list of stop words used by the TF-IDF vectorizer.

#process upload files:
This function processes the uploaded files and returns a pickled instance of the Filetb class.

The function takes two arguments:

files: A list of uploaded files.
collection (optional): A string representing the name of the collection to store the uploaded files.
The function first creates a folder to store the uploaded files. If collection is provided, it creates a sub-folder with the name of the collection in the UPLOAD_FOLDER directory, otherwise, it creates a sub-folder with the name of the current user.

It then loops through the uploaded files and saves each file to the user's folder if it has a PDF file extension.

Next, it creates an instance of the Filetb class and calls its files_processor_tb method with the list of file paths as an argument to process the files. The processed instance is then pickled and saved to a file named fp in the user's folder.

Finally, the function returns the pickled instance of the Filetb class.

# api call uplaod
The upload_files function is a FastAPI endpoint that receives a list of uploaded files as input and passes them to the process_uploaded_files function. It returns a response indicating whether the files were successfully uploaded or not. Here is the continuation of the technical documentation:

Function Signature
python
Copy code
async def upload_files(files: List[UploadFile] = File(...))
Input
files : A list of UploadFile objects, which represent the uploaded files. The UploadFile object has the following attributes:
filename : The name of the uploaded file.
content_type : The content type of the uploaded file.
file : A SpooledTemporaryFile object containing the contents of the uploaded file.
Output
The function does not return any output. It either raises an exception if an error occurs while processing the files, or returns a response indicating that the files were successfully uploaded.
Function Logic
The upload_files function first calls the process_uploaded_files function and passes the files list to it.
If the process_uploaded_files function raises an exception, the upload_files function calls the delete_files function to delete the uploaded files and re-raises the exception.
If the process_uploaded_files function completes successfully, the upload_files function returns a response indicating that the files were successfully uploaded.
Overall, the upload_files function is responsible for handling the upload of files to the server, and the process_uploaded_files function is responsible for processing the uploaded files and storing them in a format that can be used by the application.



#qnatb
This is a class called Qnatb that has methods for answering questions based on a given text. The class is initialized with a model path, which is used to load a pre-trained question answering model and tokenizer.

The answer_question method takes a question and answer text and returns the best answer based on the given question. It first preprocesses the question by removing any characters that are not alphanumeric or spaces, then adds a question mark at the end of the question. It then encodes the question and answer text using the tokenizer and passes the encoded inputs to the pre-trained model. The output of the model is the start and end logits for each token in the input text. The method finds the start and end token positions with the highest logits and extracts the corresponding answer. It also returns the start logit value and the surrounding text around the answer.

The extract_answer_blobs method takes a question and a list of response sentences and returns the best answer based on the given question. It first preprocesses the responses by splitting them into smaller chunks if they are too long. It then calls the answer_question method on each response and returns a sorted list of answers based on the start logit values.



#get final response:
    This function get_final_responses takes a qna object of class Qnatb, a question string, and an optional collection string as input. It then loads the fp object from the pickle file stored in the user folder. It calls the get_response_cosine method of the fp object to get a list of sentences as responses to the question.

Next, it calls the extract_answer_blobs method of the qna object with the question and the list of response sentences as input. This method returns a list of tuples where each tuple contains an answer, a start logit score, and the corresponding sentence from which the answer was extracted. The list of tuples is sorted based on the start logit score in descending order.

Finally, the function returns the sorted list of tuples as the result.


