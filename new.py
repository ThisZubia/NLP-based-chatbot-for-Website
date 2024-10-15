import nltk
import numpy as np
import tflearn
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

# Load the model and the data
model = keras.models.load_model("chatbot_model.tflearn")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # Tokenize and lemmatize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Create the bag of words representation
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def respond_to_query(query):
    # Predict the intent
    p = bow(query, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

def get_response(intent):
    # Fetch the response for the detected intent
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])

def chat():
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye! Have a great day!")
            break

        intents_pred = respond_to_query(user_input)
        if intents_pred:
            intent = intents_pred[0]['intent']
            response = get_response(intent)
            print("Bot:", response)

            # Follow-up questions based on the user's intent
            if intent == "help":
                follow_up = input("Bot: How can I assist you further? ")
                print("Bot: " + get_response("services"))

            elif intent == "services":
                follow_up = input("Bot: Which service are you interested in? ")
                print("Bot: Please tell me if you need help with " + follow_up)

chat()
