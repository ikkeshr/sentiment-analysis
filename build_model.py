#soruce
#Blog: https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/
#Githib: https://gist.github.com/vgpena/b1c088f3c8b8c2c65dd8edbe0eae7023
""" 
Dataset used: Twitter sentiment Analysis Dataset (contains 1,578,627 classified tweets)
Dataset Link: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/

STEPS:

1. Get the data into a usable format
2. Build the neural net
3. Train it with said data
4. Save the neural net for future use
 """

#========================================================================================================
# 1. Get the data into a usable format
#========================================================================================================


import numpy as np
import tensorflow as tf
import json


training = np.genfromtxt('Sentiment Analysis Dataset.csv ',
                              delimiter=',', skip_header=1, usecols=(1, 3),
                              dtype=None, encoding='utf8')


# create our training data from the tweets
train_x = [x[1] for x in training]
# index all the sentiment labels
train_y = np.asarray([x[0] for x in training])



# only work with the 3000 most popular words found in our dataset
max_words = 3000

# create a new Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index

# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

#Converts a text to a sequence of words (or tokens).
def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in tf.keras.preprocessing.text.text_to_word_sequence(text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)


# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = tf.keras.utils.to_categorical(train_y, 2)


#========================================================================================================
# 2. Build the neural net
#========================================================================================================

#Keras’ Sequential() is a simple type of neural net that consists of a “stack” of layers executed in order.
model = tf.keras.Sequential()

#Add layers to the Sequential neural net
model.add(tf.keras.layers.Dense(512, input_shape=(max_words,), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

#last step before training, we need to compile the network
model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])


#========================================================================================================
# 3. Train it with said data
#========================================================================================================
print("#########################################################################")

model.fit(train_x, train_y,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)


#========================================================================================================
# 4. Save the neural net for future use
#========================================================================================================

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
