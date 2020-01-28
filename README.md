# Sentiment-analysis 
Twitter is a popular social networking platform where members create and interact with messages known as “tweets”. This serves as a mean for individuals to express their thoughts and feelings about different subjects. Various different parties such as consumers and marketers have done sentiment analysis on such tweets to gather insights into products or to conduct market analysis.</br></br>

In this project, I attempt to conduct sentiment analysis on “tweets” using a simple neural network. I attempt to classify the polarity of the tweet where it is either positive or negative. For example:</br>
“I like the new design of the website!” → Positive</br>
“The new design is awful!” → Negative</br>
If the tweet has both positive and negative elements, the more dominant sentiment should be picked as the final label.</br></br>

# Installation
First make sure you have python and pip install. If you install python, pip should also be install along with it.</br></br>
Pull the code form github and run the following command in the project directory:</br>
pip install -r requirements.txt</br>
This will install all the required python packages used for the program.</br></br>

Enter your twitter APi credentials in the file twitter_api.py</br></br>

Then in the project directory run the following command:</br> python serve.py</br></br>
This will launch a flask's web server and in the console it will print the address where you need to go to access the application.</br>
