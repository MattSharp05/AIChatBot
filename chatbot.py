# Imports
import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

print(classes)

# Clean up sentence
# Take words as tokens
# Lemmatize the words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Create a bag of words
# Convert sentence into a bag of words
# List of 0s and 1s that indicate if the word is there or not
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # list will be as long as the number of words there are.
    # bag list initially consists of all 0s
    # if word is in sentence, the 0 for that word will be changed into a 1
    for w in sentence_words:
        for i, word in enumerate(words): #loop through words
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Set up a predict function
def predict_class(sentence):
    bow = bag_of_words(sentence) # Create a bag of words
    res = model.predict(np.array([bow]))[0] # Predict a result based on bag of words
    ERROR_THRESHOLD = 0.25 #To limit uncertainty
    # Enumerate through results and get index (i) and probability (r)
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True) #Sort by probability in reverse order (highest probability first)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list # return a list full of intents and probabilities


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("GO! Bot is running")
print("Hi! Welcome! I am Indigo!")
print("I am the virtual assistant for AutoNation!")
print("What can I help you with?")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)