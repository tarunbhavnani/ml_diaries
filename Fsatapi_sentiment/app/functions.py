
import numpy as np
from typing import Union



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#def get_sentiment(sent, model, tokenizer):
#    encoded_input = tokenizer(sent, return_tensors='pt')
#    output = model(**encoded_input)
#    scores = output[0][0].detach().numpy().astype("float64")
#    neg, neutral, pos = softmax(scores)
#
#    return {"negative": neg, "neutral": neutral, "positive": pos}

def get_sentiment(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return {"negative": sentiment_dict['neg'], "neutral": sentiment_dict['neu'], "positive": sentiment_dict['pos']}

