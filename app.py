from flask import Flask, render_template, request
import sentiment_analysis

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	sentiment = ''
	if request.method == 'POST':
		tweet = request.form['tweet']
		tweet = sentiment_analysis.clean_tweets(tweet)
		return render_template('result.html', sentiment = tweet)