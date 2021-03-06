import glob
import logging
import numpy as np
import os
import pickle
from stemming.porter2 import stem
import re
import string
import sys
import uuid

import nltk
import scipy
import scipy.sparse as sp

import conf
import parserFile


class PrepareMIML:
    dictionary = list()  # all the words
    labels = list()  # all the labels
    instances = list()
    matrix_instances_labels = {}
    matrix_instances_dictionary = {}

    # matrix_document_labels = np.matrix(1,2) #? utile?

    def __init__(self, use_conf):
        self.dictionary = []
        self.documents = []
        self.use_conf = use_conf
        self.STOPWORDS_FILE = "stopwords.txt"

        logging.basicConfig(level=logging.DEBUG, filename='log.txt', filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        return

    def arrayMatrixInstancesDictionary(self, filename):
        self.init_documents(filename)
        self.init_dictionary()
        array_docs = []
        self.log("Found " + str(len(self.documents)) + " documents")
        doc_excluded = []
        for i, doc in enumerate(self.documents):
            self.progress("Doc " + str(i + 1) + " of " + str(len(self.documents)))
            instances = self.sparseMatrixInstancesDictionaryOneDoc(doc['instances'])
            if instances is not None:
                array_docs.append(instances)
            else:
                doc_excluded += doc
        self.documents = [doc for doc in self.documents if doc not in doc_excluded]
        self.log("Excluded " + str(len(doc_excluded)) + " documents, now they are " + str(len(self.documents)))

        n_instances = []
        for doc in self.documents:
            n_instances.append(len(doc['instances']))
        print "Found " + str(np.sum(np.array(n_instances))) + " instances, mean is " + str(np.mean(np.array(n_instances)))

        return array_docs

    def init_documents(self, filename):
        print "Loading documents..."
        self.documents = self.read_all_files() if filename is None else self.read_file(filename)

        choice = conf.remove_docs_0_label if self.use_conf else raw_input("Want to remove documents with 0 labels? (y/n) - ")
        if choice == "y":
            self.remove_docs_0_label()

        choice = conf.remove_short_docs if self.use_conf else raw_input("Want to remove short documents? (y/n) - ")
        if choice == "y":
            self.remove_short_documents()

        choice = conf.remove_docs_1_label if self.use_conf else raw_input("Want to remove documents with only 1 label? (y/n) - ")
        if choice == "y":
            self.remove_docs_1_label()

        print "Stemming words..."
        for doc in self.documents:
            doc["words"] = [stem(word) for word in doc["words"]]

    def remove_short_documents(self):
        min_words = conf.min_words if self.use_conf else int(raw_input("Enter minimum number of words - "))

        self.log("Deleting documents with less than " + str(min_words) + " words")
        old_length = len(self.documents)
        self.documents = [doc for doc in self.documents if len(doc['words']) >= min_words]
        self.log("Removed " + str(old_length - len(self.documents)) + " short documents of " + str(old_length) + " now they are " + str(len(self.documents)))

    def remove_docs_0_label(self):
        lengths = [len(doc['labels']) for doc in self.documents]
        self.log("Deleting documents with 0 labels")
        old_length = len(self.documents)
        self.documents = [doc for doc in self.documents if len(doc['labels']) > 0]
        self.log("Removed " + str(old_length - len(self.documents)) + " useless documents of " + str(
            old_length) + " now they are " + str(len(self.documents)))

    def remove_docs_1_label(self):
        min_docs = conf.min_docs_number if self.use_conf else int(raw_input("Enter minimum number of documents - "))

        self.log("Deleting documents with 1 label")
        old_length = len(self.documents)
        bad_documents = set([doc['id'] for doc in self.documents if len(doc['labels']) == 1])
        while len(self.documents) - len(bad_documents) < min_docs and len(bad_documents) > 0:
            bad_documents.pop()

        self.documents = [doc for doc in self.documents if doc['id'] not in bad_documents]
        self.log("Removed " + str(old_length - len(self.documents)) + " documents with 1 label of " + str(
            old_length) + " now they are " + str(len(self.documents)))

    def create_dictionary(self):
        # scan all document from dataset and create the dictionary with all words
        all_words = {}
        words_index = set()

        print "Creating dictionary..."
        counter = 0
        dictionary_growth = []
        total_words = 0
        for doc in self.documents:
            total_words += len(doc['words'])

        for d, doc in enumerate(self.documents):
            doc_words = doc['words']
            dictionary_growth.append(0)
            for word in doc_words:
                counter += 1
                if counter % 1000 == 0:
                    self.progress("%.2f" % (float(100 * counter) / total_words) + "%")
                if word not in words_index:
                    dictionary_growth[d] += 1
                    words_index.add(word)
                    all_words[word] = [-1, 1]
                else:
                    all_words[word][1] += 1

        self.progress("100.00%\n")

        self.log("Found " + str(len(all_words)) + " words")

        self.dictionary = all_words

        # Remove stopwords
        choice = conf.remove_stopwords if self.use_conf else raw_input("Want to remove stopwords? (y/n) - ")
        if choice == "y":
            self.remove_stopwords()

        # Remove rare words
        choice = conf.remove_rare_words if self.use_conf else raw_input("Want to remove rare words? (y/n) - ")
        if choice == "y":
            self.remove_rare_words()

        counter = 0
        for key in self.dictionary:
            self.dictionary[key][0] = counter
            counter += 1

        with open('dictionary_stat.txt', 'w') as fp:
            pickle.dump(dictionary_growth, fp)
        with open('dictionary.txt', 'w') as fp:
            pickle.dump(self.dictionary, fp)

        return self.dictionary


    def sparseMatrixInstancesDictionaryOneDoc(self, instances):
        N = len(instances)
        M = len(self.dictionary)
        m = []
        # m *= -1
        excluded = 0;
        for i, instance in enumerate(instances):
            words = self.get_words_from_one_document(instance)
            if len(words) > 0:
                instance = np.zeros((M))
                for word in words:
                    word = stem(word)
                    if self.dictionary.has_key(word):
                        instance[self.dictionary[word][0]] += 1
                if instance.sum() > 0:
                    m.append(instance)
                else:
                    excluded += 1
        if excluded > 0:
            print "Excluded " + str(excluded) + " instances of " + str(len(instances))
        if np.asmatrix(m).sum() > 0:
            return sp.csr_matrix(np.asmatrix(m).astype(np.float32))
        else:
            return None

    def read_all_files(self):
        all_docs = []
        files = glob.glob('dataset/*.sgm')
        for f, filename in enumerate(files):
            self.progress('%.2f' % (float(100 * f) / len(files)) + '%')
            documents = self.read_file(filename)
            all_docs += documents
        self.progress('100.00%\n')
        return all_docs

    # used
    def read_file(self, filename):
        parser = parserFile.ReutersParser()
        docs = parser.parse(open(filename, 'rb'))
        return [
            {
                'id': uuid.uuid4().int,
                'text': doc[1].lower(),
                'words': re.compile('\w+').findall(doc[1].lower()),
                'instances': self.get_instances_from_text_new_version(doc[1].lower()),
                'labels': doc[0],
            }
            for doc in list(docs)
            ]



    def log(self, msg):
        # logging.info(msg)
        print msg

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

    # used
    def get_instances_from_text(self, text):
        # returns all the instances from a text document as a list (all the sentences of a text document)
        sentences = nltk.sent_tokenize(text)
        result = list(sentences)
        return result[0:len(result) - 1]

    #different way overlapping windows of 50 words
    def get_instances_from_text_new_version(self, text):
        all_words = re.compile('\w+').findall(text)
        sentences = []
        for i in range(0,len(all_words),50):
            sentences.append(string.join(all_words[i:i+50], " "))

        return sentences

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
        if not self.instances:
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

        # non funge
        # for i in range(len(instances)):
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
            for j, document in enumerate(self.read_file(filename)):
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

            for i, instance in enumerate(allinstances):
                print "Valuto instance " + str(i) + " di " + str(len(allinstances))
                for word in self.get_words_from_one_document(instance):
                    for j, word_in_dictionary in enumerate(self.dictionary):
                        if word == word_in_dictionary:
                            m[i][j] += 1

            single_data = sp.csc_matrix(m)
            scipy.io.mmwrite("mat_" + str(k) + ".mtx", single_data)
            dataset += single_data

        scipy.io.mmwrite("matrix.mtx", dataset)


    def remove_stopwords(self):
        if os.path.isfile(self.STOPWORDS_FILE):
            stopwords = []
            with open(self.STOPWORDS_FILE) as f:
                stopwords = [line.strip() for line in f.readlines() if line.strip()]
            print "May remove up to " + str(len(stopwords)) + " words"
            oldsize = len(self.dictionary)
            for word in stopwords:
                if self.dictionary.has_key(word):
                    del self.dictionary[word]
            self.log("Removed " + str(oldsize - len(self.dictionary)) + " words")
        else:
            print "There is no stopword file \"" + self.STOPWORDS_FILE + "\""

    def remove_rare_words(self):
        # self.dictionary[word][0] -> index
        # self.dictionary[word][1] -> count
        # PLOT DICTIONARY
        # word_occurrences = [data[1] for word, data in self.dictionary.iteritems()]
        # pyplot.hist(word_occurrences, bins=range(1, np.max(word_occurrences)))
        # pyplot.show()

        percent = 0
        if self.use_conf:
            percent = conf.keep_word_percent
        else:
            percent = int(raw_input("Enter percent of words - "))

        self.log("Deleting words...")
        old_length = len(self.dictionary)
        n_words = percent * len(self.dictionary) / 100
        kept_words = {}

        for i in range(n_words):
            best_word = None
            for word, data in self.dictionary.iteritems():
                if best_word is None or data[1] > self.dictionary[best_word][1]:
                    best_word = word
            kept_words[best_word] = self.dictionary[best_word]
            del self.dictionary[best_word]

        self.dictionary = kept_words
        self.log("Removed " + str(old_length - len(self.dictionary)) + " rare words, now they are " + str(len(self.dictionary)))


    # used
    def init_dictionary(self):
        create = False
        if os.path.isfile("dictionary.txt"):

            if not self.use_conf:
                choice = raw_input("Dictionary already exists, use that? (y/n) - ")
            else:
                choice = conf.keep_dictionary

            if choice == "n":
                create = True
        else:
            create = True

        # Re-create dictionary
        if create:
            self.create_dictionary()

        if (os.path.isfile("dictionary.txt")):
            with open('dictionary.txt', 'r') as fp:
                itemlist = pickle.load(fp)
            self.dictionary = itemlist
        else:
            print "Cannot create dictionary"

    def get_dictionary(self):
        with open('dictionary.txt', 'r') as fp:
            itemlist = pickle.load(fp)
        self.dictionary = itemlist
        return itemlist

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
            all_lab += self.read_all_labels_one_file(filename)
        all_lab = list(set(all_lab))
        self.labels = dict(zip(all_lab, list(xrange(len(all_lab)))))
        with open('labels.txt', 'w') as fp:
            pickle.dump(self.labels, fp)
        return all_lab

    def get_labels(self):
        with open('labels.txt', 'r') as fp:
            item_list = pickle.load(fp)
        self.labels = item_list
        return item_list

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
        self.labels = labels
        return labels

    def progress(self, str):
        print '\r' + str,
        sys.stdout.flush()

    def denseMatrixInstancesDictionaryOneDoc(self, document):
        if not self.dictionary:
            self.create_dictionary()
        instances = self.get_instances_from_text(document)
        N = len(instances)
        M = len(self.dictionary)
        m = np.zeros((N, M))
        # m *= -1
        print " instances: ", N
        for i, instance in enumerate(instances):
            words = self.get_words_from_one_document(instance)
            for word in words:
                if self.dictionary.has_key(word):
                    m[i][self.dictionary[word]] += 1
        return m


    def arrayMatrixInstancesDictionaryOneFileDense(self, filename):
        array_docs = list()
        excluded_docs = []
        docs = self.read_file(filename)
        print "Documents: ", len(docs)
        for i, doc in enumerate(docs):
            print "Doc", i, " of ", len(docs),
            instances = self.denseMatrixInstancesDictionaryOneDoc(doc[1])
            if len(instances) > 0:
                array_docs.append(instances)
            else:
                excluded_docs.append(i)
        return array_docs, excluded_docs



    def matrixDocLabelsOneFile(self):
        self.labels = {}
        all_labels = {}
        counter = 0
        first = True
        for doc in self.documents:
            for label in doc['labels']:
                #if label == 'usa': #da rimuovere
                if not self.labels.has_key(label):
                    self.labels[label] = counter
                    counter += 1
                    all_labels[label] = [-1, 1]
                else:
                    all_labels[label][1] += 1

        #we use only the 20% most frequent labels
        top_labels = {}
        counter = 0
        while len(top_labels) < conf.max_labels and len(top_labels) != len(all_labels):
            best = 0
            best_label = ""
            for label in self.labels:
                if all_labels[label][1] > best and not top_labels.has_key(label):
                    best = all_labels[label][1]
                    best_label = label
            #all_labels[best_label][0] = 1
            top_labels[best_label] = counter
            counter += 1

        self.log("Selected " + str(len(top_labels)) + " labels from " + str(len(self.labels)))
        for label in top_labels:
            self.log(label)
        self.labels = top_labels


        N = len(self.documents)
        M = len(self.labels)
        m = []

        for i, doc in enumerate(self.documents):
            labels = -1 * np.ones((M))
            for label in doc['labels']:
                if self.labels.has_key(label):
                    labels[self.labels[label]] = 1
            m.append(labels)
        return m

    def matrixDocLabels(self):
        if not self.labels:
            self.get_labels()
        N = 0
        for filename in glob.glob('dataset/*.sgm'):
            docs = self.read_file(filename)
            N += len(docs)
        M = len(self.labels)
        m = np.zeros((N, M))
        i = 0
        for filename in glob.glob('dataset/*.sgm'):
            docs = self.read_file(filename)
            for doc in docs:
                for label in doc[0]:
                    if self.labels.has_key(label):
                        m[i][self.labels[label]] += 1
                i += 1
        return m
