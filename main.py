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

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    #doc = parser.parse(open(filename, 'rb'))
    #print list(doc)
    #pprint.pprint(list(doc))

    p = prepareMIML.prepareMIML()

    p.get_full_matrix_instances_dictionary_alternative()
    #p.get_full_matrix_instances_dictionary()

    #all_labels = p.read_all_labels()
    #dictionary = p.create_dictionary()
    #print len(dictionary)

    #You have to download this data
    #nltk.download('punkt')


    #instances = p.get_all_instances()
    #print len(instances)

    #matrix = p.get_matrix_instances_labels()


    #val = p.read_file(filename)
    #val = p.read_all_documents()
    #val = p.read_all_labels_one_file(filename)

    #val = p.read_all_labels()
    #val2 = p.all_labels()

    #doc = p.get_words_one_file(filename)

    #dictionary_matrix = p.create_dictionary_matrix()

