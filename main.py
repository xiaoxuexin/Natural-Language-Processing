# coding: utf8
import numpy as np
from collections import Counter
from re import compile as _Re
import torch
import torch.nn as nn
import model
import torch.nn.functional as F
import random
_unicode_chr_splitter = _Re( '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)' ).split

random.seed(1234)

# constant parameters assigned at beginning
FREQUENCE_LIMIT = 5
FIVE_TYPE = 5
SEVEN_TYPE = 7
POEM_TYPE = 5
learning_rate = 0.1
hidden_size = 100
embedding_size = 100
num_epochs = 50
initial_index = random.randint(1,100)
word_number = 100

def split_unicode_chrs(text):
    return [chr for chr in _unicode_chr_splitter(text) if chr and chr != '〈' and chr != '〉'
            and chr != '（' and chr != '）' and chr != '￡' and chr != '○' and chr != '、' and chr != '※'
            and chr != '一']


def genrate_training_data(poem):
    while poem[-1] != '。':
        del poem[-1]
    if poem[FIVE_TYPE] == '，':
        style = FIVE_TYPE
        poetry = []
        for i in poem:
            if i != '，' and i != '。':
                poetry.append(i)
            else:
                poetry.append('END')
                poetry.append('START')
        return style, poetry
    elif poem[SEVEN_TYPE] == '，':
        style = SEVEN_TYPE
        poetry = []
        for i in poem:
            if i != '，' and i != '。':
                poetry.append(i)
            else:
                poetry.append('END')
                poetry.append('START')
        return style, poetry
    else:
        return 0, []


def load_data(filename):
    token = []
    all_five_poems = []
    all_seven_poems = []
    with open(filename, "r", encoding="utf8") as poemfile:
        first_row = next(poemfile)
        for line in poemfile.readlines()[0:500]:
            sentence = line.split()[-1]
            style, poem = genrate_training_data(split_unicode_chrs(sentence))
            if style == FIVE_TYPE:
                all_five_poems = all_five_poems + poem
            if style == SEVEN_TYPE:
                all_seven_poems = all_seven_poems + poem
            token = token + split_unicode_chrs(sentence)
        counter = Counter(token)
        counter['START'] = FREQUENCE_LIMIT + 1
        counter['END'] = FREQUENCE_LIMIT + 1
        unique_word_list = [i for i in counter if counter[i] > FREQUENCE_LIMIT and i != '，' and i != '。']
        unique_word_list.append('UNK')
        for i in range(len(all_five_poems)):
            if counter[all_five_poems[i]] <= FREQUENCE_LIMIT:
                all_five_poems[i] = 'UNK'
        for i in range(len(all_seven_poems)):
            if counter[all_seven_poems[i]] <= FREQUENCE_LIMIT:
                all_seven_poems[i] = 'UNK'
        return unique_word_list, all_five_poems, all_seven_poems




def i2ohv(word_list):
    index_dict = {}
    vector_dict = {}
    # chara_dict = {}
    for i in range(len(word_list)):
        index_dict[word_list[i]] = i
        zero_vec = np.zeros((len(word_list), 1))
        zero_vec[i] = 1
        vector_dict[i] = zero_vec
    return index_dict, vector_dict


def load_embedding(vocab_size, embedding_size):
    return nn.Embedding(vocab_size, embedding_size)


def train(input_list, unique_word_list):
    vocab_size = len(unique_word_list)
    word_embedding = load_embedding(vocab_size, embedding_size)
    rnn = model.RNN(embedding_size, hidden_size, vocab_size, word_embedding)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rnn.to(device)

    hidden = rnn.initHidden()

    # rnn.zero_grad()

    # for i in input_list:
    #     print(unique_word_list[i])

    criterion = nn.CrossEntropyLoss()
    loss = 0
    optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

    for epoch in range(1, 1 + num_epochs):
        print("epoch is ", epoch)
        for i in range(len(input_list) - 1):

            # word_embedding = load_embedding(vocab_size, embedding_size)
            input_ind = torch.tensor(input_list[i])
            input = word_embedding(input_ind)

            optimizer.zero_grad()

            output, hidden_var = rnn(input, hidden)
            # print(type(output), output.size())
            # new_output = word_embedding(output[0].max(0)[1])

            target = torch.tensor(input_list[i + 1])
            # target = torch.reshape(word_embedding(target_ind), (1, embedding_size))

            target = torch.reshape(target, (1,))
            # output = torch.reshape(output)

            # print(target)
            # print(output)
            # print(type(target), target.shape)
            # print(type(output), output.size())

            loss = criterion(output, target)
            # l = nn.CrossEntropyLoss(new_output, target, reduction='none')

            # loss += l

            # loss.sum().backward(retain_graph=True)
            loss.backward()

            optimizer.step()

            # for p in rnn.parameters():
            #     p.data.add_(-learning_rate, p.grad.data)
        print("end of epoch")

    rnn.eval()

    i = 0
    poem = []
    index = torch.tensor(initial_index)

    while i < word_number:
        # print(i)
        outcome, hidden_var1 = rnn(word_embedding(index), hidden_var)

        # print(list(outcome))

        prediction = outcome[0].max(0)[1]
        # print(prediction)
        prediction_character = unique_word_list[prediction.numpy()]
        # print('character ', prediction_character)

        # if prediction_character != 'END' and prediction_character != 'UNK' and prediction_character != 'START':
        poem.append(prediction_character)
        i = i + 1
        index = prediction
        # else:
        #     index = torch.randint(0, vocab_size, (1,))
        #     print("index is ", index)
        #         # print(unique_word_list[index.numpy()])
        #     hidden_var = hidden_var + torch.rand(hidden_var.size())


        # rnn.train()

    # for i in [0, POEM_TYPE, 2 * POEM_TYPE, 3 * POEM_TYPE]:
    #     if i == 0 or i == 2 * POEM_TYPE:
    #         print(''.join(poem[i: i + POEM_TYPE]) + '，')
    #     else:
    #         print(''.join(poem[i: i + POEM_TYPE]) + '。')

    return poem


if __name__=='__main__':
    unique_word_list, all_five_poems, all_seven_poems = load_data('qtais_tab.txt')
    # print(all_seven_poems)
    index_dict, vector_dict = i2ohv(unique_word_list)
    input_list = []

    for i in range(len(all_five_poems)):
        input_list.append(index_dict[all_five_poems[i]])

    poem = train(input_list, unique_word_list)
    print(poem)
