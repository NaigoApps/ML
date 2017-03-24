import re

import scipy

import parserFile
import glob
import numpy as np
import nltk
import scipy.sparse as sp


class prepareMIML:
    dictionary = list()  # all the words
    labels = list()  # all the labels
    instances = list()
    matrix_instances_labels = {}
    matrix_instances_dictionary = {}

    # matrix_document_labels = np.matrix(1,2) #? utile?

    def __init__(self, encoding='latin-1'):
        return

    def get_all_instances(self):
        # returns all the instances from all files in the dataset (all the sentences)
        all_instances = list()
        for filename in glob.glob('dataset/*.sgm'):
            all_instances += self.get_instances_from_file(filename)
        self.instances = all_instances
        return all_instances

    def get_instances_from_file(self, filename):
        # returns all the instances from a file (with more texts) as a list (all the sentences of a file)
        parsed_file = self.read_file(filename)
        instances = list()
        for document in parsed_file:
            instances += self.get_instances_from_text(document[1])
        return instances

    def get_instances_from_text(self, text):
        # returns all the instances from a text document as a list (all the sentences of a text document)
        sentences = nltk.sent_tokenize(text)
        return list(sentences)

    def get_matrix_instances_labels(self):
        # returns the matrix where in the row there are all the instances (sentences) of a document
        # and the columns are all the labels. In a cell [instance][label] there is 1 if exists a document with the sentence "instance" and labeled with "label"
        # if the matrix already exists, return this

        if self.matrix_instances_labels:
            return self.matrix_instances_labels

        matrix = {}
        for filename in glob.glob('dataset/*.sgm'):
            parsed_file = self.read_file(filename)
            for document in parsed_file:
                # create an associative matrix with [instance,label] = 1 foreach instance and label in the document
                for instance in self.get_instances_from_text(document[1]):
                    for label in document[0]:
                        matrix[instance][label] = 1
        self.matrix_instances_labels = matrix
        return matrix

    def get_matrix_instances_dictionary(self):
        # returns the matrix where in the row there are all the instances (sentences) of a document
        # and the columns are all the labels. In a cell [instance][word] there is 1 if the instance contains the word

        if self.matrix_instances_dictionary:
            return self.matrix_instances_dictionary
        if (not self.instances):
            self.get_all_instances()
        # if(not self.dictionary):
        #    self.create_dictionary()

        matrix = {}
        for instance in self.instances:
            for word in self.get_words_from_one_document(instance):
                matrix[instance][word] = 1
                # for words in self.dictionary:
                #    if str(instance).__contains__(words):
                #        matrix[instance][words] = 1
        self.matrix_instances_dictionary = matrix
        return matrix

    def get_matrix_instances_dictionary_one_document(self, document, matrix_name=""):
        # returns the matrix where in the row there are all the instances (sentences) of a document
        # and the columns are all the labels. In a cell [instance][word] there is 1 if the instance contains the word

        if not self.dictionary:
            self.create_dictionary()

        instances = self.get_instances_from_text(document)
        N = len(instances)
        M = len(self.dictionary)
        m = np.zeros((N, M))

        for i in range(len(instances)):
            for words in self.get_words_from_one_document(instances[i]):
                for j in range(len(self.dictionary)):
                    if words == self.dictionary[j]:
                        m[i][j] += 1

        #non funge
        #for i in range(len(instances)):
        #    for words in self.get_words_from_one_document(instances[i]):
        #        find = self.dictionary.index(words)
        #        if find:
        #            m[i][find] += 1

        matrix = sp.csc_matrix(m)
        self.matrix_instances_dictionary = matrix
        return matrix

    def get_full_matrix_instances_dictionary(self):
        # returns the matrix where in the row there are all the instances (sentences) of a document
        # and the columns are all the labels. In a cell [instance][word] there is 1 if the instance contains the word

        if self.matrix_instances_dictionary:
            return self.matrix_instances_dictionary

        dataset = list()
        i = 0
        for filename in glob.glob('dataset/*.sgm'):
            i += 1
            print "Elaborazione file: " + filename
            for j,document in enumerate(self.read_file(filename)):
                print "Doc " + str(j)
                single_data = self.get_matrix_instances_dictionary_one_document(document[1])
                dataset += single_data

            scipy.io.mmwrite("mat_" + str(i) + ".mtx", single_data)
        scipy.io.mmwrite("matrix.mtx", dataset)


    def get_full_matrix_instances_dictionary_alternative(self):
        # returns the matrix where in the row there are all the instances (sentences) of a document
        # and the columns are all the labels. In a cell [instance][word] there is 1 if the instance contains the word

        if self.matrix_instances_dictionary:
            return self.matrix_instances_dictionary
        if not self.dictionary:
            self.create_dictionary()

        dataset = list()
        k = 0
        for filename in glob.glob('dataset/*.sgm'):
            k += 1
            print "Elaborazione file: " + filename
            allinstances = self.get_instances_from_file(filename)

            N = len(allinstances)
            M = len(self.dictionary)
            m = np.zeros((N, M))

            for i,instance in enumerate(allinstances):
                print "Valuto instance " + str(i) + " di " + str(len(allinstances))
                for word in self.get_words_from_one_document(instance):
                    for j,word_in_dictionary in enumerate(self.dictionary):
                        if word == word_in_dictionary:
                            m[i][j] += 1

            single_data = sp.csc_matrix(m)
            scipy.io.mmwrite("mat_" + str(k) + ".mtx", single_data)
            dataset += single_data

        scipy.io.mmwrite("matrix.mtx", dataset)

    def create_dictionary(self):
        # scan all document from dataset and create the dictionary with all words
        print "Creating dictionary..."
        all_words = set()
        docs = self.read_all_files()
        for doc in docs:
            all_words.update(self.get_words_from_one_document(doc[1]))
        self.dictionary = list(all_words)
        return list(all_words)

    def get_words_from_file(self, filename):
        words = list()
        doc = self.read_file(filename)
        for texts in doc:
            words += self.get_words_from_one_document(texts[1])
        return list(set(words))

    def get_words_from_one_document(self, doc):
        return re.compile('\w+').findall(doc)

    def read_all_labels(self):
        # read and returns all label from dataset
        all_lab = list()
        for filename in glob.glob('dataset/*.sgm'):
            all_lab = all_lab + self.read_all_labels_one_file(filename)
        self.all_labels = list(set(all_lab))
        return list(set(all_lab))

    def read_all_labels_one_file(self, filename):
        # read and returns all label from a document
        all_labels = list()
        doc = self.read_file(filename)
        for texts in doc:
            all_labels = all_labels + texts[0]
        # remove all duplicated
        return list(set(all_labels))

    def all_labels_complete(self):
        labels = list()
        file = open("dataset/all-exchanges-strings.lc.txt")
        for line in file:
            labels.append(line)
        file = open("dataset/all-orgs-strings.lc.txt")
        for line in file:
            labels.append(line)
        file = open("dataset/all-people-strings.lc.txt")
        for line in file:
            labels.append(line)
        file = open("dataset/all-places-strings.lc.txt")
        for line in file:
            labels.append(line)
        file = open("dataset/all-topics-strings.lc.txt")
        for line in file:
            labels.append(line)
        self.all_labels = labels
        return labels

    def read_all_files(self):
        all_docs = list()
        for filename in glob.glob('dataset/*.sgm'):
            all_docs += self.read_file(filename)
        return all_docs

    def read_file(self, filename):
        parser = parserFile.ReutersParser()
        doc = parser.parse(open(filename, 'rb'))
        return list(doc)


    def sparseMatrixInstancesDictionaryOneDoc(self, document):
        if not self.dictionary:
            self.create_dictionary()
        instances = self.get_instances_from_text(document)
        N = len(instances)
        M = len(self.dictionary)
        m = np.zeros((N, M))
        print " instances: ",N
        for i, instance in enumerate(instances):
            words = self.get_words_from_one_document(instance)
            for word in words:
                for j, word_in_dictionary in enumerate(self.dictionary):
                    if word == word_in_dictionary:
                        m[i][j] += 1
        sparse_matrix = sp.csc_matrix(m)
        #return sparse_matrix
        return m

    def arrayMatrixInstancesDictionaryOneFile(self, filename):
        array_docs = list()
        docs = self.read_file(filename)
        print "Documents: ",len(docs)
        for i, doc in enumerate(docs):
            print "Doc", i, " of ", len(docs),
            array_docs.append(self.sparseMatrixInstancesDictionaryOneDoc(doc[1]))
        return array_docs

    def arrayMatrixInstancesDictionary(self):
        array_docs = list()
        for filename in glob.glob('dataset/*.sgm'):
            array_docs += self.arrayMatrixInstancesDictionaryOneFile(self, filename)
        return array_docs
