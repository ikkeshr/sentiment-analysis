from flask import Flask, render_template, jsonify, request
from twitter_api import Twitter
from model import Model
import numpy as np

app = Flask(__name__)

@app.route("/")
def serve():
    return render_template('index.html')

@app.route('/submit_query')
def submit_query():
    print ('Received...')
   
    query = request.args.get('query')
    tweet_count = request.args.get('tweet_count')
    req = "request received by flask!! Request was " + query + ", fetch " + tweet_count + " tweets"
    print (req)

    twitter = Twitter()
    tweets = twitter.fetch_tweets(query, int(tweet_count))

    labels = ['negative', 'positive']
    model = Model()

    results = []
    for tweet in tweets:
        pred = model.predict(tweet)
        result = {
            "tweet": tweet,
            "sentiment": labels[np.argmax(pred)],
            "confidence": round(pred[0][np.argmax(pred)] * 100, 1)
        }
        results.append(result)

    return jsonify(results)


@app.route('/submit_text')
def submit_text():
    print ('Received...')
   
    sentence = request.args.get('sentence')
    print ("request received by flask!!")

    labels = ['negative', 'positive']
    model = Model()

    pred = model.predict(sentence)

    results = {
        "sentiment": labels[np.argmax(pred)],
        "confidence": round(pred[0][np.argmax(pred)] * 100, 1)
    }
    #results = "This is a %s sentiment, I am %d%% sure." % (labels[np.argmax(pred)], round(pred[0][np.argmax(pred)] * 100, 1))
    return jsonify(results)
    

if __name__ == "__main__":
    app.run(debug=True)