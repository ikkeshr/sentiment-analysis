import tweepy
import re

class Twitter:
    def __init__(self):
        #Enter your twitter credential here
        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_token_secret = ''

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth,wait_on_rate_limit=True)

    def clean_tweet(self, tweet):
        return (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)"," ",tweet).split()))

    def fetch_tweets(self, query, limit):
        tweets = []
        print ('Fetching tweets...')
        for tweet in tweepy.Cursor(self.api.search,q=query, #result_type="popular",
                                    count=limit,lang="en",
                                    since="2020-01-01").items(limit):
            tweets.append(tweet.text)
        return tweets