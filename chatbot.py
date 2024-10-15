import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
from tensorflow import keras
import pickle

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Preprocess the data
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'s", "'"]

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents in the corpus
        documents.append((word_list, intent['tag']))
        # Add to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and sort the words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the processed data
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag + output_row)

# Shuffle and split the data into training and testing sets
random.shuffle(training)
training = np.array(training)

train_x = training[:, :-len(classes)]
train_y = training[:, -len(classes):]

# Build the model
model = tflearn.DNN(tflearn.input_data(shape=[None, len(train_x[0])]))
model.fit(train_x, train_y, n_epoch=200, batch_size=8, show_metric=True)

# Save the trained model
model.save("chatbot_model.tflearn")
