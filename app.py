from flask import Flask, request, render_template
import spacy
import pickle
import numpy as np
import gensim.downloader as api

# Load the spaCy model and the Word2Vec model
nlp = spacy.load("en_core_web_lg")
w2v = api.load("word2vec-google-news-300")

# Load the trained classifier
with open('GRBOOSINGnews.pkl', 'rb') as f:
    model = pickle.load(f)

def preandvec(text):
    doc = nlp(text)
    filtered = [word.lemma_ for word in doc if not word.is_stop and not word.is_punct]
    return w2v.get_mean_vector(filtered)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    category_map = {0: 'tech', 1: 'business', 2: 'sport', 3: 'politics', 4: 'entertainment'}
    try:
        vector = preandvec(text).reshape(1, -1)
        prediction = model.predict(vector)[0]
        label = category_map[prediction]
        output = f'Category: {label}'
        return render_template('index.html', prediction_text=output, text=text)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', text=text)

if __name__ == "__main__":
    app.run(debug=True)
