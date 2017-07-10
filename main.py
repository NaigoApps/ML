# def transform(mx):
#     """
#     Transform into np.matrix if array/list
#     ignore scipy.sparse matrix
#     """
#     if issparse(mx):
#         return mx.todense()
#     return np.asmatrix(mx)

import numpy as np
import random
import sys

import scipy.sparse as sp

import miml_svm
import prepareMIML
from LearningResult import LearningResult

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

def extract_randomly(data,labels, percent):
    res_data, res_labels = [], []
    size = int(len(data) * percent / 100)
    for index in range(0, size):
        rand = random.randint(0, len(data) - 1)
        res_data.append(data.pop(rand))
        res_labels.append(labels.pop(rand))
    return res_data, res_labels

def symmetric_difference(a, b):
    sum = 0
    for i in range(len(a)):
        sum += 1 if a[i]*b[i] < 0 else 0
    return sum

def hamming_loss(prediction, actual):
    sum = 0
    for i in range(len(prediction)):
        sum += 1.0/len(prediction[i])*symmetric_difference(prediction[i], actual[i])
    return sum / len(prediction)

def one_error(predictions, actual):
    sum = 0.0
    for i in range(len(predictions)):
        best = predictions[i][0]
        i_best = 0
        for l in range(len(predictions[i])):
            if predictions[i][l] > best:
                best = predictions[i][l]
                i_best = l
        if predictions[i][i_best] * actual[i][i_best] < 0:
            sum += 1
    return sum / len(predictions)

def make_ranks(v):
    v = np.array(v)
    ranks = np.zeros(len(v))
    cur_rank = 1
    for i in range(len(v)):
        ranks[np.argmax(v)] = cur_rank
        cur_rank += 1
        v[np.argmax(v)] = v[np.argmin(v)] - 1
    return ranks


def coverage(predictions, actuals):
    sum = 0.0
    for i in range(len(predictions)):
        ranks = make_ranks(predictions[i])
        pos_indexes = []
        for j in range(len(actuals[i])):
            if actuals[i][j] > 0:
                pos_indexes.append(j)
        if len(pos_indexes) > 0:
            best_ranked_index = pos_indexes[0]
            for j in pos_indexes:
                if ranks[j] > ranks[best_ranked_index]:
                    best_ranked_index = j
            sum += ranks[best_ranked_index] - 1

    return sum / len(predictions)

def rank_loss(predictions, actuals):
    sum = 0.0
    for p in range(len(predictions)):
        cur_sum = 0.0
        positives = 0
        for i in range(len(actuals[p])):
            if actuals[p][i] > 0:
                positives += 1
            for j in range(len(actuals[p])):
                #foreach (i,j) where i is a real label and j is not
                if actuals[p][j] < 0 < actuals[p][i]:
                    if predictions[p][i] < predictions[p][j]:
                        cur_sum += 1
        if positives * (len(actuals[p]) - positives) > 0:
            cur_sum /= positives * (len(actuals[p]) - positives)
            sum += cur_sum
    return sum / len(predictions)

def avg_precision(predictions, actuals):
    sum = 0.0
    for i in range(len(predictions)):
        ranks = make_ranks(predictions[i])
        pos_indexes = []
        for y in range(len(actuals[i])):
            if actuals[i][y] > 0:
                pos_indexes.append(y)
        for y in pos_indexes:
            cur_sum = 0.0
            for y1 in pos_indexes:
                if ranks[y1] <= ranks[y]:
                    cur_sum += 1
            cur_sum /= ranks[y]
        if len(pos_indexes) > 0:
            sum += cur_sum / len(pos_indexes)
    return sum / len(predictions)

def merge_results(values):
    return np.average(values), np.std(values)

if __name__ == "__main__":

    if (len(sys.argv) == 2):
        config_file = sys.argv[1]
    else:
        config_file = None
    p = prepareMIML.PrepareMIML(config_file)

    dataset = p.arrayMatrixInstancesDictionary('./dataset/reut2-000.sgm')
    # dataset = p.arrayMatrixInstancesDictionary(None)

    # print dense_matrix[3] #matrice instanza dizionario del documento 3
    # print dense_matrix[3][2] #dizionario dell'instanza 2 del doc 3
    # print dense_matrix[3][2][1] #parola 1 dell'instanza 2 del documento 3
    # print "Words second doc, first phrase"
    # print sum(dense_matrix[2][1]) #numero di parola nell'instanza 1 doc 2
    # print ""

    labels = p.matrixDocLabelsOneFile()

    for label in np.array(labels).transpose():
        sum = 0.0
        for doc in label:
            if doc > 0:
                sum += 1
        print sum
    # print labels_matrix #matrice documento label
    # print labels_matrix[0] #tutte le label del doc 1
    # print labels_matrix[0][1] #label 1 del documento 0
    # print "Labels first doc"
    # print sum(labels_matrix[0]) #numero di labels del doc 0

    # solo se matrice densa
    # print len(result) #numero di documenti (1000 solo nel primo file)
    # print len(result[0]) #numero di instanze del documento 0 -> 21
    # print len(result[0][0]) #numero di parole del dizionario -> 48377
    # print sum(result[0][0]) #numero di parole nell'instanza 0 -> 38


    # save_sparse_csc('array_of_matrix_doc_dictionary_file1',result[0])
    # matrix = load_sparse_csc('array_of_matrix_doc_dictionary_file1.npz')
    # matrix.todense()

    # p.create_dictionary()
    # result = p.matrixDocLabels()

    # for d, document in enumerate(dense_matrix):
    #     dense_matrix[d] = sp.dok_matrix(document)
    # for i, instance in enumerate(document):
    #     document[i] = sp.dok_matrix(instance)


    hlosses = []
    coverages = []
    oneerrors = []
    rlosses = []
    avg_precs = []
    avg_recs = []
    avg_F1s = []

    times = 2

    for tries in range(times):
        training_set = list(dataset)
        training_labels = list(labels)
        test_set, test_labels = extract_randomly(training_set, training_labels, 25)

        svm = miml_svm.MiMlSVM()
        svm.train(training_set, training_labels)
        predictions = svm.test(test_set)

        result = LearningResult(predictions, test_labels)

        print "Hloss... ", result.hamming_loss()
        print "Oneerror... ", result.one_error()
        print "Coverage... ", result.coverage()
        print "Rank loss... ", result.ranking_loss()
        print "Avg precision... ", result.average_precision()
        print "Avg recall... ", result.average_recall()
        print "Avg F1... ", result.average_F1()
        hlosses.append(result.hamming_loss())
        oneerrors.append(result.one_error())
        coverages.append(result.coverage())
        rlosses.append(result.ranking_loss())
        avg_precs.append(result.average_precision())
        avg_recs.append(result.average_recall())
        avg_F1s.append(result.average_F1())

        # true_negatives = 0
        # true_positives = 0
        # false_negatives = 0
        # false_positives = 0
        # for i, prediction in enumerate(predictions):
        #     for j, predicted_label in enumerate(prediction):
        #
        #         if predicted_label < 0 and labels[i][j] < 0:
        #             true_negatives += 1
        #         if predicted_label > 0 and labels[i][j] > 0:
        #             true_positives += 1
        #         if predicted_label < 0 < labels[i][j]:
        #             false_negatives += 1
        #         if labels[i][j] < 0 < predicted_label:
        #             false_positives += 1
        # print "True positives: ", true_positives
        # print "True negatives: ", true_negatives
        # print "False positives: ", false_positives
        # print "False negatives: ", false_negatives
        # accuracy = float(true_negatives + true_positives) \
        #            / (true_negatives + true_positives + false_negatives + false_positives + 1)
        # precision = float(true_positives) / (true_positives + false_positives + 1)
        # recall = float(true_positives) / (true_positives + false_negatives + 1)
        # print "Accuracy: ", accuracy
        # print "Precision: ", precision
        # print "Recall: ", recall

        # accuracies.append(accuracy)
        # precisions.append(precision)
        # recalls.append(recall)

    hloss_mean, hloss_sd = merge_results(hlosses)
    oneerror_mean, oneerror_sd = merge_results(oneerrors)
    coverage_mean, coverage_sd = merge_results(coverages)
    rloss_mean, rloss_sd = merge_results(rlosses)
    avg_prec_mean, avg_prec_sd = merge_results(avg_precs)
    avg_rec_mean, avg_rec_sd = merge_results(avg_recs)
    avg_F1_mean, avg_F1_sd = merge_results(avg_F1s)

    print "Hloss : ", hloss_mean, " +/- " , hloss_sd
    print "Oneerror : ", oneerror_mean, " +/- " , oneerror_sd
    print "Coverage : ", coverage_mean, " +/- " , coverage_sd
    print "Rloss : ", rloss_mean, " +/- " , rloss_sd
    print "AVGPrecision : ", avg_prec_mean, " +/- " , avg_prec_sd
    print "AVGRecall : ", avg_rec_mean, " +/- " , avg_F1_mean
    print "AVGF1 : ", avg_F1_mean, " +/- " , avg_F1_sd

    # plt.figure()
    # plt.title("H-loss")
    # plt.plot(range(0, len(hlosses)), hlosses)
    # plt.show()
    # plt.figure()
    # plt.title("Precision")
    # plt.plot(range(0, len(precisions)), precisions)
    # plt.show()
    # plt.figure()
    # plt.title("Recall")
    # plt.plot(range(0, len(recalls)), recalls)
    # plt.show()
