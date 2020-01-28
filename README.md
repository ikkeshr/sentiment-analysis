# Sentiment-analysis 
This is a web app that allows search twitter and it will perform sentiment analysis on the resulting tweets of the search query. You can also enter a text to perform sentiment analysis on. The sentiment analysis is done by a Neural Network model built with the Keras API present in the tensorflow python's library. The model was train with the Sentiment140 dataset and it yields an accuracy of 77%.

# Installation
First make sure you have python and pip install. If you install python, pip should also be install along with it.</br></br>
Pull the code form github and run the following command in the project directory:</br>
pip install -r requirements.txt</br>
This will install all the required python packages used for the program.</br></br>

Enter your twitter APi credentials in the file twitter_api.py</br></br>

Then in the project directory run the following command:</br> python serve.py</br></br>
This will launch a flask's web server and in the console it will print the address where you need to go to access the application.</br>
