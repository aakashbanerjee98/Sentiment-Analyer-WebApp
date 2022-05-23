from flask import Flask
from flask import Flask, render_template, request
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from langdetect import detect
import numpy as np


app = Flask(__name__)
with open('tokenizer3.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model=load_model('checkpoint.h5')

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    text1=request.form['rooms']
    a=1
    try:
        a=detect(text1)
    except :
        return render_template('index.html', prediction_text=f'You have not entered the text in bengali'.upper())

    if a=='bn':
        tw = tokenizer.texts_to_sequences([text1])
        tw = pad_sequences(tw,maxlen=45)
        prediction = model.predict(tw)
        y1=[]
        ans=""
        for i in prediction:
            y1.append(np.argmax(i))
        if y1==[1]:
            ans="Negative"
        else:
            ans="Positive"

        return render_template('index.html', prediction_text=f'Predicted label: {ans}')
    else:
        return render_template('index.html', prediction_text=f'You have not entered the text in bengali'.upper())

if __name__ == "__main__":
    app.run(debug=True)