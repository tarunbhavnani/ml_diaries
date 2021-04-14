from flask import Flask, request, jsonify
from flask_restful import Api
import torch
import logging
logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


app= Flask(__name__)
api = Api(app) #Flask REST Api code 
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\Bert-qa\model')
def answer_question(question, answer_text):
    encoded_dict = tokenizer.encode_plus(text=question,text_pair=answer_text, add_special=True)
    input_ids = encoded_dict['input_ids']
    segment_ids = encoded_dict['token_type_ids']
    output = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    answer_start = torch.argmax(output['start_logits'])
    start_logit= output['start_logits'][0][answer_start].detach().numpy()
    answer_end = torch.argmax(output['end_logits'])
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer, start_logit

#text_blob="""Roger Federer (German pronunciation: [ˈrɔdʒər ˈfeːdərər]; born 8 August 1981) is a Swiss professional tennis player. He is ranked No. 7 in the world by the Association of Tennis Professionals (ATP). He has won 20 Grand Slam men's singles titles, an all-time record shared with Rafael Nadal. Federer has been world No. 1 in the ATP rankings a total of 310 weeks – including a record 237 consecutive weeks – and has finished as the year-end No. 1 five times. Federer has won 103 ATP singles titles, the second-most of all-time behind Jimmy Connors and including a record six ATP Finals."""
@app.route('/predict',methods=["POST"])
def predict():
    if request.method=="POST":
        file1 = request.files['file']
        sentence=file1.read().decode("utf-8")
        file2 = request.files['text_blob']
        text_blob=file2.read().decode("utf-8")
        print(sentence)
        answer,_=answer_question(sentence, text_blob)
        #answer,_=answer_question("who is federer", text_blob)
        return jsonify({'answer': answer})

@app.route('/')
def hello():
    return "predictio is happening on /predict"



if __name__ == '__main__':
    app.run(port= 3995)