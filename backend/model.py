# importing the required libraries
import random 
import json 
import pickle 
import numpy as np
import tensorflow as tf 
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

# initializing the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopWords = nltk.corpus.stopwords.words('english')

# loading the intents file
intents = json.load(open('intents.json', 'r'))

# initializing the required variables
all_words = []   
tags = []
documents = []
ignore_letters = [',','.','!',"$",'?']

# function to tokenize the words
def tokernize(token_word):
    return nltk.word_tokenize(token_word) # break down texts into small units
# function to stem the words
def stem(word):
    return lemmatizer.lemmatize(word.lower()) # convert the word into root form

# looping through the intents file
# intents is the json file which act as a list of dictionaries. Each dictionary contains the tag, patterns and responses
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    # intents file eke intents kiyn list eke thisyen dictionaries wla "tag" key ekt adala values tika tags kiyn list ekt append krnwa. 

    for pattern in intent['patterns']: 
        wordlist = tokernize(pattern)
        all_words.extend(wordlist)
        documents.append((wordlist, tag))
        # inner loop ekk dala thiyenne e tag ekt adal dictionary ekenm "patterns" key ekt adala values access krnn ona nisa. ewa tikanize krla all_words
        # kiyn list ekt extend krnwa. append krn nethuw extend krnne, ekm list ekk widiyt gnn ona nisa.
        # eetpasse tokenize krpu patterns tika (word_list) ekt adala tag ekath ekk pair krla document kiyn list ekt append krnwa

# stemming the words and removing the stopwords
all_words = [stem(word) for word in all_words if word not in ignore_letters] # all_words list eke thiyn words, ignore_letter kiyn list eke neththn root word ekt convert krnwa
all_words = sorted(set(all_words)) 
tags = sorted(set(tags))
# sorted()--> sort the list in ascending order, set()--> remove the duplicates

# saving the words and tags
pickle.dump(all_words, open('words.pkl', 'wb')) 
pickle.dump(tags, open('classes.pkl', 'wb'))
# pickle.dump()---> save the data to a file  'wb'---> write binary (means write the data in binary format)

# creating the training data
training = []
outputEmpty = [0] * len(tags) # creates a list of zeros used as a template for creating one-hot encoded output labels.
# one-hot encoding ekdi tag ekkt adala okkom elements 0 wla thiyenne, eka lable ekkt adala elemets 1 wlata thiyenne

# looping through the documents
for  (document, tag)  in documents:
    bag = []
    wordPatterns = document

    wordPatterns = [stem(word) for word in wordPatterns]
    for word in all_words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)

    outputRow[tags.index(tag)] = 1
    training.append(bag + outputRow)

# shuffling the training data
random.shuffle(training)

# converting the training data to numpy array
training = np.array(training)

# splitting the data into trainX and trainY
trainX = training[:, :len(all_words)]
trainY = training[:, len(all_words):]

# creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])
# relu --------> introduces non-linearity to the model and helps it learn complex patterns.
# softmax ----> converts the output layer into a probability distribution.

# optimizer configuration
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# model compilation
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# early stopping
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')

# training the model
model.fit(np.array(trainX), np.array(trainY), epochs=500, batch_size=5, verbose=1, callbacks=[earlyStopping])

# saving the model
model.save('chatbot_model.keras')
print('Done')

# evaluating the model
model_accurcy = model.evaluate(np.array(trainX), np.array(trainY))
print("⭕⭕⭕ Model accuracy: ", model_accurcy[1] * 100, "%⭕⭕⭕")