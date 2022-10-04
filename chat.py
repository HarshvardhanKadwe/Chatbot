import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#to check if we can use cuda for faster or better performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Neur"
print("Let's chat! (type 'quit' to exit)")

count=0

while True:
    #sentence from the user
    sentence = input("You: ")
    if sentence == "quit":
        break
    
    #Chatbot asks the user "Do you like apples or oranges?" within the first 5 interactions
    if count==5:
        print(f"{bot_name}: Do you like apples or oranges?")

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    #so we try and find the output with max probability
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    #the threshhold is defined by the developer
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                count+=1
    else:
        print(f"{bot_name}: I do not understand...")