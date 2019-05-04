import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
import json


#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.

#... (4) Test your model. Compare cosine similarities between learned word vectors.










#.................................................................................
#... global variables
#.................................................................................
# from word2vec import origcounts

random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = []                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from






#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................



def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = False
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load(open("w2v_wordcodes.p","rb"))
        uniqueWords = pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    # ... load in first 15,000 rows of unlabeled data file.  You can load in
    # more if you want later (and should do this for the final homework)
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    fullconts = fullconts  # (TASK) Use all the data for the final submission
    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]




    print ("Generating token stream...")
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.
    stop_words = set(stopwords.words('english'))
    fullrec = nltk.word_tokenize(fullconts[0])
    fullrec = [w for w in fullrec if not w in stop_words]
    min_count = 50
    origcounts = Counter(fullrec)





    print ("Performing minimum thresholding..")
    #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)


    for i in origcounts.keys():
        if origcounts[i] >= min_count:
            wordcounts[i] = origcounts[i]

    fullrec_filtered = []
    for i in range(len(fullrec)):
        if origcounts[fullrec[i]] < min_count:
            fullrec_filtered.append("UNK")
        else:
            fullrec_filtered.append(fullrec[i])



    print ("Producing one-hot indicies")
    #... (TASK) sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = list(wordcounts.keys()) #... fill in
    uniqueWords.append("UNK")
    for i in range(len(uniqueWords)):
        wordcodes[uniqueWords[i]] = i #... fill in

    # print(fullrec)
    # print(fullrec_filtered)
    for i in range(len(fullrec)):
        # if fullrec_filtered[i] != "UNK":
        fullrec_filtered[i] = wordcodes[fullrec_filtered[i]]  # ... fill in
    # print(fullrec_filtered)
    # ... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered





    #... close input file handle
    handle.close()



    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.
    #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.

    # print(fullrec)
    return fullrec







#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit
def sigmoid(x):
    return 1.0/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    #global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0
    exp_count_list = []

    print ("Generating exponentiated count vectors")
    #... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    for i in range(len(uniqueWords) - 1):
        exp_count_list.append(wordcounts[uniqueWords[i]] ** (-exp_power)) #... fill in
    exp_count_array = np.array(exp_count_list)
    max_exp_count = sum(exp_count_array)



    print ("Generating distribution")

    #... (TASK) compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = exp_count_array / max_exp_count #... fill in





    print ("Filling up sampling table")
    #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.
    cumulative_dict = {} # ... fill in
    table_size = 1e7
    cumulative_dist = prob_dist * table_size
    # print(cumulative_dist)
    summ = 0
    for i in range(len(cumulative_dist)):
        summ = summ + cumulative_dist[i]
        cumulative_dist[i] = summ
    index = 0
    for i in range(len(prob_dist)):
        while index < cumulative_dist[i]:
            cumulative_dict[index] = i
            index = index + 1
    # print(cumulative_dist)
    # print(cumulative_dict)
    return cumulative_dict






#.................................................................................
#... generate a specific number of negative samples
#.................................................................................


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    i = 0
    while i < num_samples:
        rand_num = random.randint(0, len(samplingTable.keys()) - 1)
        # print(rand_num)
        # print(context_idx)
        if rand_num != context_idx:
            results.append(samplingTable[rand_num])
            i = i + 1
        # print("generate sample " + str(i))
    return results




# @jit(nopython=True)
# def performDescent(num_samples, learning_rate, center_token, sequence_chars,W1,W2,negative_indices):
#     # sequence chars was generated from the mapped sequence in the core code
#     nll_new = 0
#     for k in range(0, len(sequence_chars)):
#         #... (TASK) implement gradient descent. Find the current context token from sequence_chars
#         #... and the associated negative samples from negative_indices. Run gradient descent on both
#         #... weight matrices W1 and W2.
#         #... compute the total negative log-likelihood and store this in nll_new.
#
#
#
#
#     return [nll_new]









#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0




    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            if iternum%10000==0:
                print ("Negative likelihood: ", nll)
                nll_results.append(nll)
                nll = 0


            #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence

            center_token = mapped_sequence[i] #... fill in
            #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            if center_token == "UNK":
                continue


            iternum += 1

            #... now propagate to each of the context outputs
            for k in range(0, len(context_window)):


                neg_sum = 0
                #... (TASK) Use context_window to find one-hot index of the current context token.
                context_index = mapped_sequence[i + context_window[k]] #... fill in


                #... construct some negative samples
                negative_indices = generateSamples(context_index, num_samples)

                #... (TASK) You have your context token and your negative samples.
                #... Perform gradient descent on both weight matrices.
                #... Also keep track of the negative log-likelihood in variable nll.
                # print("center token " + str(center_token))
                # print("context ind" + str(context_index))
                x1 = sigmoid(np.matmul(W2[context_index, :], np.transpose(W1[center_token, :]))) - 1
                ssum = x1 * W2[context_index, :]
                W2[context_index, :] = W2[context_index, :] - learning_rate * (x1) * W1[center_token, :]

                for nind in negative_indices:

                    x2 = sigmoid(np.matmul(W2[nind, :], np.transpose(W1[center_token, :])))
                    neg_sum = neg_sum + np.log(sigmoid(-np.matmul(W2[nind, :], np.transpose(W1[center_token, :]))))
                    ssum = ssum + x2 * W2[nind, :]
                    W2[nind, :] = W2[nind, :] - learning_rate * (x2) * W1[center_token, :]
                W1[center_token, :] = W1[center_token, :] - learning_rate * ssum
                nll = -neg_sum - np.log(x1 + 1)


    return [W1,W2]



#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
    handle = open("saved_W1.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()






#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.






#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    print("first time training")
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    print("save the weights")
    save_model(word_embeddings, proj_embeddings)









#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [word_seq[0], # suffix averaged
    embeddings[wordcodes[word_seq[1]]]]
    vector_math = vectors[0]+vectors[1]
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def get_neighbors(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    dict = {}
    #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
    post_train = np.matmul(proj_embeddings, np.transpose(word_embeddings))
    target_vector = post_train[:, target_word]
    for i in range(len(uniqueWords)):
        outputs.append(1 - cosine(target_vector, post_train[:, i]))
    output_arr = np.array(outputs)
    ind = np.argpartition(output_arr, -11)[-11:]
    top_ind = ind[np.argsort(output_arr[ind])]
    top_val = output_arr[top_ind]
    for i in range(9, -1, -1):
        dict[uniqueWords[top_ind[i]]] = top_val[i]
    return dict







if __name__ == '__main__':
    # if len(sys.argv)==2:
    #     filename = sys.argv[1]
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData('unlabeled-data.txt')
        print ("Full sequence loaded...")




        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        train_vectors(preload=False)
        [word_embeddings, proj_embeddings] = load_model()


        #... we've got the trained weight matrices. Now we can do some predictions
        targets = ["good", "bad", "scary", "popular", "different", "right", "new", "american", "music", "wikipedia"]
        for targ in targets:
            bestpreds = get_neighbors(wordcodes[targ])
            with open("prob7-output.txt", "a") as text_file:
                print("Target: ", targ, file= text_file)
                text_file.write(json.dumps(bestpreds))
                text_file.write('\n')


        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        # print (analogy(["son", "daughter", "man"]))
        # print (analogy(["thousand", "thousands", "hundred"]))
        # print (analogy(["amusing", "fun", "scary"]))
        # print (analogy(["terrible", "bad", "amazing"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.

        # s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
        # others = [["types", "type"],
        #           ["ships", "ship"],
        #           ["values", "value"],
        #           ["walls", "wall"],
        #           ["spoilers", "spoiler"]]
        # for rec in others:
        #     s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
        # s_suffix = np.mean(s_suffix, axis=0)
        # print (morphology([s_suffix, "techniques"]))
        # print (morphology([s_suffix, "sons"]))
        # print (morphology([s_suffix, "secrets"]))
        # test_file_list = []
        # with open('intrinsic-test.tsv', 'r') as tsvfile:
        #     tsvin = csv.reader(tsvfile, delimiter='\t')
        #     first_row = next(tsvin)
        #     for row in tsvin:
        #         if row[1] not in test_file_list:
        #             test_file_list.append(row[1])
        #         if row[2] not in test_file_list:
        #             test_file_list.append(row[2])

        with open('intrinsic-test.tsv', 'r') as tsvin, open('output.csv', 'w') as csvout:
            tsvin = csv.reader(tsvin, delimiter = '\t')
            first_row = next(tsvin)
            csvout = csv.writer(csvout)
            csvout.writerow(['id', 'sim'])
            for row in tsvin:
                if row[1] not in uniqueWords:
                    row[1] = 'UNK'
                ind1 = wordcodes[row[1]]
                if row[2] not in uniqueWords:
                    row[2] = 'UNK'
                ind2 = wordcodes[row[2]]
                vec1 = np.matmul(proj_embeddings, word_embeddings[ind1, :])
                vec2 = np.matmul(proj_embeddings, word_embeddings[ind2, :])
                sim = 1 - cosine(vec1, vec2)
                csvout.writerow([row[0], sim])
    # else:
    #     print ("Please provide a valid input filename")
    #     sys.exit()


