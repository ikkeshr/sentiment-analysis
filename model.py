import json
import tensorflow as tf
import re

class Model:
    def __init__(self):
        dictionary_path = 'sentiment-analysis-model/dictionary.json'
        model_path = 'sentiment-analysis-model/model.json'
        model_weights_path = 'sentiment-analysis-model/model.h5'

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=3000)
        
        self.load_dictionary(dictionary_path)
        self.load_model(model_path, model_weights_path)



    # this utility makes sure that all the words in the input
    # are registered in the dictionary
    # before trying to turn them into a matrix.
    def convert_text_to_index_array(self, text):
        words = tf.keras.preprocessing.text.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in self.dictionary:
                wordIndices.append(self.dictionary[word])
            else:
                print("'%s' not in training corpus; ignoring." %(word))
        return wordIndices


    def load_dictionary(self, path):
        # read in our saved dictionary
        with open(path, 'r') as dictionary_file:
            self.dictionary = json.load(dictionary_file)


    def load_model(self, model_path, weight_path):
        # read in the saved model structure
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # and create a model from that
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        # and weight your nodes with your saved values
        self.model.load_weights(weight_path)

    def clean_text(self, text):
        return (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)"," ",text).split()))

    def remove_hastags(self, text):
        return re.sub("\s([#][\w_-]+)","",text)
    
    def predict(self, text):
        #remove symbols, hashtags etc from text
        text = self.clean_text(self.remove_hastags(text))

        # format the input for the neural net
        testArr = self.convert_text_to_index_array(text)
        model_input = self.tokenizer.sequences_to_matrix([testArr], mode='binary')

        # predict which bucket the input belongs in
        pred = self.model.predict(model_input)

        return pred