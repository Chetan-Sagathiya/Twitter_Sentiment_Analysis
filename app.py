from flask import Flask, render_template, request
import spacy
import pandas as pd
import numpy as np
import pickle

import en_core_web_sm
nlp = en_core_web_sm.load()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)

model = pickle.load(open("random_forest.pkl", "rb"))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	sentiment = ''
	if request.method == 'POST':
		tweet = request.form['tweet']
		tweet = str(clean_tweets(tweet))
		print(" *************************************** ", tweet)
		vec = nlp(tweet).vector
		vec = vec.reshape(1, -1)
		print("**************", vec, vec.shape, type(vec))
		result = model.predict(vec)
		print("result is", result)
		return render_template('result.html',)


wordnet = WordNetLemmatizer()
def clean_tweets(tweet):
	tempArr = []
	tweet = re.sub(r"[^a-zA-Z]+", ' ', tweet)
	tweet = tweet.lower()
	tweet = tweet.split()
	tweet = [wordnet.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
	if len(tweet) == 0:
	  # if all the words im tweet are stopwords then
	  tweet = 'I'
	else:
	  tweet = ' '.join(tweet)
	tweet
	tempArr.append(tweet)
	return tempArr