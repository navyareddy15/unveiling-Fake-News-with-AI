from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    return text

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"

def manual_testing(news):
    new_def_test = pd.DataFrame({"text": [news]})
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = LR.predict(new_xv_test)
    return output_label(pred_lr[0])

# Load a smaller subset of the dataset for testing
true = pd.read_csv('True.csv').sample(n=3000, random_state=42)
fake = pd.read_csv('Fake.csv').sample(n=1000, random_state=42)

true['label'] = 1
fake['label'] = 0

news = pd.concat([fake, true], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1)
news = news.sample(frac=1).reset_index(drop=True)
news['text'] = news['text'].apply(wordopt)

x = news['text']
y = news['label']

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x)

LR = LogisticRegression()
LR.fit(xv_train, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    news_article = request.form['news_article']
    result = manual_testing(news_article)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
