import time

import parserFile
import prepareMIML
import nltk
import nltk.data

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
    result = list()
    outputfile = open('array_densematrix_file1.txt', 'w')
    p.create_dictionary()
    result = p.arrayMatrixInstancesDictionaryOneFile(filename)
    print >> outputfile, result