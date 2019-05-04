import numpy as np
import json
from collections import Counter
import torch
import torch.nn as nn

LOW_FREQ = 2
epoches = 5
hidden_size = 100

def better_tokenize(sentence):
    new_toke = []
    bet_toke = sentence.split(" ")
    for i in range(len(bet_toke)):
        bet_toke[i] = ''.join(char for char in bet_toke[i] if char.isalpha())
        bet_toke[i] = bet_toke[i].lower()
        new_toke.append(bet_toke[i])
    return new_toke


def load_embedding(vocab_size, embedding_size):
    return nn.Embedding(vocab_size, embedding_size)


def load_data(file_name):
    poem_train_data = []
    with open(file_name) as file:
        for line in file.readlines()[0:100]:
            sentence = json.loads(line)['s']
            # print(sentence)
            poem_train_data.append(sentence)
            # print(poem_train_data)
            # poem_train_data.append('END')
            # poem_train_data.append('START')

    print(poem_train_data)
    pre_counter = Counter(poem_train_data)

    for i in range(len(poem_train_data)):
        if pre_counter[poem_train_data[i]] < LOW_FREQ:
            poem_train_data[i] = 'UNK'
    # print(poem_train_data)
    counter = Counter(poem_train_data)
    unique_word_list = list(counter.keys())
    # print(unique_word_list)
    unique_index_dict = {}
    for i in range(len(unique_word_list)):
        key = unique_word_list[i]
        unique_index_dict[key] = i

    return poem_train_data, unique_word_list, unique_index_dict


def train(train_data, unique_word, unique_ind_dic):
    vocab_size = len(unique_word)
    word_embedding = load_embedding(vocab_size, hidden_size)

    for epoch in range(1, epoches + 1):
        for i in range(len(train_data) - 1):
            input_index = torch.tensor(unique_ind_dic[train_data[i]])
            input = word_embedding(input_index)


if __name__=='__main__':
    training_poem, unique_word, unique_index_dict = load_data('lines.json')
    train(training_poem,unique_word, unique_index_dict)



