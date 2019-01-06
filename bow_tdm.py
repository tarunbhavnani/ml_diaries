##tfidf
##tdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
'This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?']

X = vectorizer.fit_transform(corpus)
dt=X.toarray()
df=pd.DataFrame(data=dt, columns=vectorizer.get_feature_names())


##############3
#1st approach--use tdm/bow
import numpy as np
dt=np.asarray(dt)
dt.shape
c1=[1,1,0,1]
#get dummies for labels as that is what keras will understand
c2 = pd.get_dummies(c1).values
idx=pd.get_dummies(c1).columns
#c2df=pd.DataFrame(data=c2, columns=idx)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=dt.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
#model.fit(X, Y, epochs=150, batch_size=10)

model.fit(x=dt,y=c2,epochs=5, validation_split=.25)
model.predict(dt)
prd=pd.DataFrame(data=np.round(model.predict(dt)),columns=idx)




#2nd approach use padding

corpus

w2i={}
#w2i["-PAD-"]=0
#w2i["-OOV-"]=1
counter=0
for i in corpus:
    for j in i.split():
        if j not in w2i:
            w2i[j]=counter
            counter+=1
         

w2v=[]
for i in corpus:
    sq=[]
    for j in i.split():
        if j in w2i:
            sq.append(w2i[j])
        else:
            sq.append("-OOV-")
    w2v.append(sq)
        

ln=[len(i.split()) for i in corpus]
max_len=max(ln)

from keras.preprocessing.sequence import pad_sequences


w2v_pad=pad_sequences(w2v,maxlen=max_len)
inp=max([max(i) for i in w2v])

model2=Sequential()
from keras.layers import Embedding
model2.add(Embedding(input_dim=inp+1,output_dim=128,input_length=w2v_pad.shape[1]))
from keras.layers import SpatialDropout1D
model2.add(SpatialDropout1D(rate=.1))
model2.summary()
from keras.layers import LSTM
model2.add(LSTM(units=300, dropout=.1, recurrent_dropout=.1))
model2.add(Dense(2,activation='softmax'))
model2.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

model2.fit(x=w2v_pad,y=c2, epochs=5)

model2.predict(w2v_pad)
prd2=pd.DataFrame(data=np.round(model2.predict(w2v_pad)),columns=idx)
"""
#notes
dense=1 is for continous, then you dont hot encode the y.
lstm cant be used in the non pad one, since the inputs are not in sequence
they are much like a normal deep learning algo where
every word depicts a variable and we are solvig for an equation.
"""
