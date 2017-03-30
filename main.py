import pickle
import numpy as np
import scipy.sparse as sp
from sys import getsizeof
import time

import parserFile
import prepareMIML
import nltk
import nltk.data


def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


if __name__ == "__main__":
    # Open the first Reuters data set and create the parser
    filename = "dataset/reut2-000.sgm"
    filename2 = "dataset/reut2-001.sgm"
    parser = parserFile.ReutersParser()

    #ORIGINAL
    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    #doc = parser.parse(open(filename, 'rb'))
    #print list(doc)
    #pprint.pprint(list(doc))

    p = prepareMIML.prepareMIML()

    #MATRIX
    #create the matrix instances-dictionary foreach document and foreach file
        #p.get_full_matrix_instances_dictionary_alternative()
        #p.get_full_matrix_instances_dictionary()

    #READ FILE
    #return a matrix where rows are documents (of ONE file) and columns are two: the first with labels and the second with the text
        #val = p.read_file(filename)
    #return a matrix as above but from ALL files
        #val = p.read_all_files()

    #DICTIONARY
    #return the complete dictionary (all words in ALL files) len: 48377
        #dictionary = p.create_dictionary()
    #return all the words from a FILE
        #words = p.get_words_from_file(filename)
    #return all the words from a TEXT
        #words = p.get_words_from_one_document(document)

    #INSTANCES
    #return all instances from ONE file
        #instances = p.get_instances_from_file(filename)
    #return all instances (all sentences from ALL files) len: 123432
        #instances = p.get_all_instances()
    #return all instances from a TEXT
        #instances = p.get_instances_from_text(text)

    #LABELS
    #return all the labels from ONE file
        #labels = p.read_all_labels_one_file(filename)
    #return all the labels from ALL files
        #all_labels = p.read_all_labels()


    # You have to download this data
    # nltk.download('punkt')

    #PROVA
    # val = p.read_file(filename)
    # matrix = p.matrixInstancesDictionaryOneDoc(val[0][1])

    #p.matrixDocLabels() #create the matrix Documents-Labels
    p.get_dictionary_2()
    #sparse_matrix = p.arrayMatrixInstancesDictionaryOneFile(filename)
    dense_matrix = p.arrayMatrixInstancesDictionaryOneFileDense(filename)

    print dense_matrix[3] #matrice instanza dizionario del documento 3
    print dense_matrix[3][2] #dizionario dell'instanza 2 del doc 3
    print dense_matrix[3][2][1] #parola 1 dell'instanza 2 del documento 3
    print sum(dense_matrix[2][1]) #numero di parola nell'instanza 1 doc 2
    print ""

    labels_matrix = p.matrixDocLabelsOneFile(filename)
    print labels_matrix #matrice documento label
    print labels_matrix[0] #tutte le label del doc 1
    print labels_matrix[0][1] #label 1 del documento 0
    print sum(labels_matrix[0]) #numero di labels del doc 0

    #solo se matrice densa
    # print len(result) #numero di documenti (1000 solo nel primo file)
    # print len(result[0]) #numero di instanze del documento 0 -> 21
    # print len(result[0][0]) #numero di parole del dizionario -> 48377
    # print sum(result[0][0]) #numero di parole nell'instanza 0 -> 38


    #save_sparse_csc('array_of_matrix_doc_dictionary_file1',result[0])
    #matrix = load_sparse_csc('array_of_matrix_doc_dictionary_file1.npz')
    #matrix.todense()


    # p.create_dictionary()
    # result = p.matrixDocLabels()


