# Imports
import random
import json
import pickle
import numpy as np


# Download the NLTK Tokens
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Lemmatize the words
lemmatizer = WordNetLemmatizer()

# Load json file
intents = json.loads(open('intents.json').read())


# Extracting Data
words = []
classes = []
documents = []
ignore_lettters = ['?', '!', '.', ',']


# Load Training Data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #takes string and splits it up into individual words
        words.extend(word_list)
        documents.append((word_list, intent["tag"])) # tags words with classification/group
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#print(documents)

# Prepare training data

# Lemmatize the words
# Lemmatization is the process of converting a word to its base form.
# lemmatization considers the context and converts the word to its meaningful base form
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_lettters]
words = sorted(set(words)) # Sort list and remove duplicates
#print(words)

classes = sorted(set(classes))

# Save the data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Convert data into numerical values
training = []
output_empty = [0] * len(classes)

# Create a bag of words
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: # Check if word occours in pattern
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data and put into array
random.shuffle(training)
training = np.array(training)

# Split into x and y values
train_x = list(training[:,0]) # Everything in 0 dimension
train_y = list(training[:,1]) # Everything in 1st dimension


#Build Neural Network Model
model = Sequential()
#Add input layer, dense layer with 128 neurons, input shape dependant on training data size for x
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
# Add another dense layer with 64 neurons
model.add(Dense(64, activation='relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
# Add another dense layer with as many neurons as there are classes
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print('Done')