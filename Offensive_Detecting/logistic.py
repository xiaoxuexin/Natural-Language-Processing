from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import csv
from random import randint
import numpy as np

ignore = ["the", "a", "an", "you", "i", "he", "she", "and", "or", "they", "it", "is", "are", "am", "be",
          "myself", "himself", "herself"]


def better_tokenize(sentence):
    new_toke = []
    bet_toke = sentence.split(" ")
    for i in range(len(bet_toke)):
        bet_toke[i] = ''.join(char for char in bet_toke[i] if char.isalpha())
        bet_toke[i] = bet_toke[i].lower()
        if (bet_toke[i] in ignore) == False:
            new_toke.append(bet_toke[i])
    return new_toke


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(new_vector, beta):
    prob = sigmoid(np.matmul(new_vector, beta)[0])
    print(prob)
    if prob > 0.5:
        return 1
    else:
        return 0


def log_likelihood(x, y, beta):
    return np.matmul(x, beta)[0] * y - np.log(1 + np.exp(np.matmul(x, beta)[0]))


def compute_gradient(x, y, beta):
    return (y - sigmoid(np.matmul(x, beta)[0])) * x


learning_rate = [5e-6, 5e-5, 0.01]
step_num = 50000
row_ind = 0
col_ind = 0
row_num = 0
feature_counter = Counter()
label_list = []
with open('train.csv', 'r') as read_train:
    read_train_file = csv.reader(read_train, delimiter =',')
    first_row = next(read_train_file)
    for row in read_train_file:
        label_list.append(int(row[0]))
        row_num = row_num + 1
        feature_counter = feature_counter + Counter(better_tokenize(row[1]))

count = 0
feature_list = list(feature_counter.keys())
feature_num = len(feature_list)
matrix = np.zeros((row_num, feature_num))
column_diction = {}
label_array = np.array(label_list).reshape(row_num, 1)
for i in feature_list:
    count += 1
    column_diction[i] = count

with open('train.csv', 'r') as read_train:
    read_train_file = csv.reader(read_train, delimiter =',')
    first_row = next(read_train_file)
    for row in read_train_file:
        temp_count = Counter(better_tokenize(row[1]))
        for i in temp_count.keys():
            col_ind = column_diction[i] - 1
            matrix[row_ind][col_ind] = temp_count[i]
        row_ind += 1

X = np.concatenate((matrix, np.ones((row_num, 1))), 1)


def logistic_regression(X, Y, learning_rate, num_step):
    i = 0
    theta_plot = []
    plot_x = []
    log_likelihood_list = []
    theta0 = np.zeros((feature_num + 1, 1))
    while i < num_step:
        random_num = randint(0, row_num - 1)
        theta1 = theta0 + learning_rate * compute_gradient(X[random_num, :], Y[random_num][0], theta0).reshape((feature_num + 1, 1))
        diff = abs(theta1 - theta0)
        theta0 = theta1
        i = i + 1
        if i % 500 == 0:
            plot_x.append(i)
            log_likelihood_list.append(log_likelihood(X[random_num, :], Y[random_num][0], theta0))
            theta_plot.append(sum(diff)[0])
    return theta0, log_likelihood_list, plot_x


thetan, log_likelihood_list1, plot_x1 = logistic_regression(X, label_array, learning_rate[0], step_num)
thetann, log_likelihood_list2, plot_x2 = logistic_regression(X, label_array, learning_rate[1], step_num)
thetannn, log_likelihood_list3, plot_x3 = logistic_regression(X, label_array, learning_rate[2], step_num)
plt.plot(plot_x1, log_likelihood_list1, label='learning rate 5e-6')
plt.plot(plot_x2, log_likelihood_list2, label='learning rate 5e-5')
plt.plot(plot_x3, log_likelihood_list3, label='learning rate 0.01')
plt.xlabel('Step Number')
plt.ylabel('log-likelihood var')
plt.title('plot of log-likehood with diff learning rate')
plt.legend()
plt.show()

with open('logistic-outputs.csv', 'w') as nb_output:
    nb_file = csv.writer(nb_output, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    nb_file.writerow(['Insult', 'Comment'])


with open('test.csv', 'r') as read_test:
    read_test_file = csv.reader(read_test, delimiter = ',')
    first_row = next(read_test_file)
    for row in read_test_file:
        test_word = Counter(better_tokenize(row[1]))
        t = np.zeros((1, (feature_num + 1)))
        t[0][-1] = 1
        for word in test_word.keys():
            if word in feature_list:
                ind = feature_list.index(word)
                t[0][ind] = test_word[word]
        with open('logistic-outputs.csv', 'a') as log_output:
            nb_file = csv.writer(log_output, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            nb_file.writerow([predict(t, thetannn), row[1]])


dev_label_list = []
dev_prediction_list = []
with open('dev.csv', 'r') as read_development:
    read_dev_file = csv.reader(read_development, delimiter = ',')
    first_row_dev = next(read_dev_file)
    for row in read_dev_file:
        dev_label_list.append(int(row[0]))
        dev_word = Counter(better_tokenize(row[1]))
        tdev = np.zeros((1, (feature_num + 1)))
        tdev[0][-1] = 1
        for word in dev_word.keys():
            if word in feature_list:
                ind = feature_list.index(word)
                tdev[0][ind] = dev_word[word]
        dev_prediction_list.append(predict(tdev, thetannn))
print(f1_score(dev_label_list, dev_prediction_list))