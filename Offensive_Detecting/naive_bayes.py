from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import csv

alpha = [x * 0.1 for x in range(1, 30)]
performance = []
Counter_0 = Counter()  # initial counter of words not insult
Counter_1 = Counter()  # initial counter of words insult
ignore = ["the", "a", "an", "you", "i", "he", "she", "and", "they", "it", "is",
          "are", "am", "be", "myself", "himself", "herself"]
num_ins = 0
num_not_ins = 0


def tokenize(sentence):
    toke = sentence.split(" ")
    return toke


def better_tokenize(sentence):
    new_toke = []
    bet_toke = sentence.split(" ")
    for i in range(len(bet_toke)):
        bet_toke[i] = ''.join(char for char in bet_toke[i] if char.isalpha())
        bet_toke[i] = bet_toke[i].lower()
        if (bet_toke[i] in ignore) == False:
            new_toke.append(bet_toke[i])
    return new_toke


# with open('better-naive-bayes-outputs.csv', 'w') as nb_output:
with open('naive-bayes-outputs.csv', 'w') as nb_output:
    nb_file = csv.writer(nb_output, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    nb_file.writerow(['Insult', 'Comment'])


with open('train.csv', 'r') as read_train:
    read_train_file = csv.reader(read_train, delimiter =',')
    first_row = next(read_train_file)
    for row in read_train_file:
        if row[0] == "0":
            num_not_ins += 1
            Counter_0 = Counter_0 + Counter(tokenize(row[1]))
            # Counter_0 = Counter_0 + Counter(better_tokenize(row[1]))
        else:
            num_ins += 1
            Counter_1 = Counter_1 + Counter(tokenize(row[1]))
            # Counter_1 = Counter_1 + Counter(better_tokenize(row[1]))


counter_total = Counter_1 + Counter_0
d = len(counter_total.keys())
num_total = num_ins + num_not_ins
prob_insult = num_ins / num_total
prob_not_insult = num_not_ins / num_total
# word in class 1 but not in class 0
word_not0_in1 = Counter(Counter_1.keys() - Counter_0.keys())
for i in word_not0_in1.keys():
    word_not0_in1[i] = 0
need_smooth_0 = Counter_0 + word_not0_in1  # class 0 need to be smoothed
# word in class 0 but not in class 1
word_not1_in0 = Counter(Counter_0.keys() - Counter_1.keys())
for i in word_not1_in0.keys():
    word_not1_in0[i] = 0
need_smooth_1 = Counter_1 + word_not1_in0  # class 1 need to be smoothed


def train(word, smoothing_alpha=0.0):
    if word in need_smooth_0:
        smooth_prob_0 = (need_smooth_0[word] + smoothing_alpha) / (sum(Counter_0.values()) + smoothing_alpha * d)
    else:
        smooth_prob_0 = 1
    if word in need_smooth_1:
        smooth_prob_1 = (need_smooth_1[word] + smoothing_alpha) / (sum(Counter_1.values()) + smoothing_alpha * d)
    else:
        smooth_prob_1 = 1
    return smooth_prob_0, smooth_prob_1


def classify(toke_sentence, temp_alpha):
    class_prob_insult = 1
    class_prob_not_insult = 1
    prob_0 = 1
    prob_1 = 1
    for i in toke_sentence:
        cond_prob_0, cond_prob_1 = train(i, temp_alpha)
        prob_0 = class_prob_not_insult * cond_prob_0
        prob_1 = class_prob_insult * cond_prob_1
    class_prob_insult = prob_1 * prob_insult
    class_prob_not_insult = prob_0 * prob_not_insult
    if class_prob_not_insult > class_prob_insult:
        return 0
    else:
        return 1


# choose an optimal alpha that has the best performance on development data
# find the alpha by plotting the errors of different alpha
for i in range(len(alpha)):
    print(alpha[i])
    prediction_dev = []
    true_classification = []
    with open('dev.csv', 'r') as read_development:
        read_dev_file = csv.reader(read_development, delimiter = ',')
        first_row_dev = next(read_dev_file)
        for row in read_dev_file:
            true_classification.append(int(row[0]))
            # prediction_dev.append(classify(better_tokenize(row[1]), alpha[i]))
            prediction_dev.append(classify(tokenize(row[1]), alpha[i]))
    performance.append(f1_score(true_classification, prediction_dev))

f1_max_ind = performance.index(max(performance))
optimal_alpha = alpha[f1_max_ind]
# print(performance)
print(optimal_alpha)

with open('test.csv', 'r') as read_test:
    read_test_file = csv.reader(read_test, delimiter = ',')
    first_row = next(read_test_file)
    for row in read_test_file:
        pred_classify = classify(tokenize(row[1]), optimal_alpha)
        # pred_classify = classify(better_tokenize(row[1]), optimal_alpha)
        # with open('better-naive-bayes-outputs.csv', 'a') as nb_output:
        with open('naive-bayes-outputs.csv', 'a') as nb_output:
            nb_file = csv.writer(nb_output, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            nb_file.writerow([pred_classify, row[1]])

plt.plot(alpha, performance)
plt.show()