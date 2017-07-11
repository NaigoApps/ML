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
from sklearn import metrics
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

def merge_results(values):
    return np.average(values), np.std(values)

def zero_based(labels):
    result = np.zeros((len(labels), len(labels[0])))
    for (r, prediction) in enumerate(labels):
        for (c, label) in enumerate(prediction):
            if label > 0:
                result[r][c] = 1
            else:
                result[r][c] = 0
    return result

if __name__ == "__main__":

    if (len(sys.argv) == 2):
        config_file = sys.argv[1]
    else:
        config_file = None
    p = prepareMIML.PrepareMIML(config_file)

    # dataset = p.arrayMatrixInstancesDictionary('./dataset/reut2-000.sgm')
    dataset = p.arrayMatrixInstancesDictionary(None)

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
        print "Sklearn coverage -> ", metrics.coverage_error(zero_based(test_labels), predictions)
        print "Rank loss... ", result.ranking_loss()
        print "Sklearn rank loss -> ", metrics.label_ranking_loss(zero_based(test_labels), predictions)
        print "Avg precision... ", result.average_precision()
        print "Sklearn avg prec -> ", metrics.label_ranking_average_precision_score(zero_based(test_labels), predictions)
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
