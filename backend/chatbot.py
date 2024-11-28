# importing the necessary libraries
import random
import json
import pickle
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer


# Load the data
intents = json.load(open('intents.json', 'r'))
words = pickle.load(open('words.pkl', 'rb'))
tags = pickle.load(open('classes.pkl', 'rb'))

# Load the model
model = tf.keras.models.load_model('chatbot_model.keras')

lemmatizer = WordNetLemmatizer()

# cleaning up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# creating the bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):  # enumerate()--> returns the index and the value of all the items in the list
            if word == w:                 
                bag[i] = 1
    return np.array(bag)

# predicting the tag of the sentence
def predict_tag (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': tags [r[0]], 'probability': str(r[1])})
    return return_list

# getting the response
def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                if float(intents_list[0]['probability']) < 0.5:
                    return "I'm sorry. I don't understand that. Please try again."
                return random.choice(i['responses'])
    except IndexError:
        return "I'm sorry. I don't understand that. Please try again."

# starting the chat
def start_chat(message):
    ints = predict_tag(message)
    res = get_response(ints, intents)
    return res

# def run_chat():
#     print("Start talking with the bot (type quit to stop)!")
#     while True:
#         message = input("You: ")
#         if message.lower() == "quit":
#             break
#         ints = predict_tag(message)
#         res = get_response(ints, intents)
#         print("Bot:", res)


# if __name__ == "__main__":
#     run_chat()


