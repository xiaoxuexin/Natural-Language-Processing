from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np
import json
import random

random.seed(1234)

embedding_size = 100
first_output_dim = 150
second_output_dim = 100
num_epo = 250
drop_rate = 0.2
word_number = 100
num_sentence = 3

def load_data(file_name):
    poem_train_data = []
    with open(file_name) as file:
        for line in file.readlines()[1000:4000]:
            sentence = json.loads(line)['s'].lower()
            # print(sentence)
            poem_train_data.append(sentence)

    return poem_train_data


def create_model(predictors, label, max_seq_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, embedding_size, input_length=max_seq_len - 1))

    model.add(LSTM(first_output_dim, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(LSTM(second_output_dim))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=num_epo, verbose=1, callbacks=[earlystop])

    return model


def generate_text(model, seed_text, next_words, max_sequ_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequ_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose = 0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


corpus = load_data('lines.json')
print(corpus)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# print(total_words)

input_sequence = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i+1]
        input_sequence.append(n_gram_sequence)

print("=====================================")
# print(input_sequence)

max_sequence_len = max([len(x) for x in input_sequence])
input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))
print(input_sequence)

predictors, label = input_sequence[:,:-1], input_sequence[:,-1]
label = ku.to_categorical(label, num_classes=total_words)
# print(predictors)
# print(label)

model = create_model(predictors, label, max_sequence_len, total_words)
poem = generate_text(model, "Meow meow meow", word_number, max_sequence_len)

print(poem)
# poem = poem.split(' ')
# leng = len(poem)
# ind = int(leng/num_sentence)
# poem.insert(ind, '\n')
# poem.insert(2*ind + 1, '\n')
# poem = ' '.join(poem)
# print(poem)
with open("xiaoxuex-poem.txt", "a") as text_file:
    text_file.write(poem)