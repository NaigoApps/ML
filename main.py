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
import scipy.io as sio

import miml_svm
import parserFile
import prepareMIML


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



def merge_results(values):
    return np.average(values), np.std(values)

if __name__ == "__main__":

    if (len(sys.argv) == 2):
        config_file = sys.argv[1]
    else:
        config_file = None
    p = prepareMIML.PrepareMIML(config_file)

    # dataset = p.arrayMatrixInstancesDictionary('./dataset/reut2-000.sgm')
    dataset = p.arrayMatrixInstancesDictionary('./dataset/reut2-000.sgm')

    # print dense_matrix[3] #matrice instanza dizionario del documento 3
    # print dense_matrix[3][2] #dizionario dell'instanza 2 del doc 3
    # print dense_matrix[3][2][1] #parola 1 dell'instanza 2 del documento 3
    # print "Words second doc, first phrase"
    # print sum(dense_matrix[2][1]) #numero di parola nell'instanza 1 doc 2
    # print ""

    labels = p.matrixDocLabelsOneFile()
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


    accuracies = []
    precisions = []
    recalls = []
    hlosses = []
    oneerrors = []

    times = 2

    for tries in range(times):
        training_set = list(dataset)
        training_labels = list(labels)
        test_set, test_labels = extract_randomly(training_set, training_labels, 25)

        svm = miml_svm.MiMlSVM()
        svm.train(training_set, training_labels)
        predictions = svm.test(test_set)

        hloss = hamming_loss(np.sign(predictions), test_labels)
        oneerror = one_error(predictions, test_labels)
        print "Hloss... ", hloss
        print "Oneerror... ", oneerror
        hlosses.append(hloss)
        oneerrors.append(oneerror)

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

    print "Hloss : ", hloss_mean, " +/- " , hloss_sd
    print "Oneerror : ", oneerror_mean, " +/- " , oneerror_sd

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
