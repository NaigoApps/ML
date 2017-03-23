import time

import parserFile
import prepareMIML
import nltk
import nltk.data

if __name__ == "__main__":
    filename = "dataset/reut2-000.sgm"
    filename2 = "dataset/reut2-001.sgm"
    parser = parserFile.ReutersParser()

    p = prepareMIML.prepareMIML()

    all_dataset = p.get_full_matrix_instances_dictionary_alternative();


    # dataset[0] -> documento
    # dataset[0][0] -> prima instance del documento
    # dataset[0][0][0] = 1 se la prima parola del dizionario è presente in dataset[0][0]
    #
    # labels[0] -> labels del documento 0
    # labels[0][0] -> 1 se label 0 è nel documento 0