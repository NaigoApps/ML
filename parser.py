'''
pM = prepareMIML.prepareMIML()

#dictionary = create_dictionary(documents)
dictionary = pM.create_dictionary()

#all_labels = read_all_labels()
all_labels = pM.read_all_labels()

all_documents = read_all_documents()

foreach document : all_documents
    document.instances = create_instances(document.text)
    foreach instance : document.instances
        instance.mega_vector = create_mega_vector(dictionary, instance.words)
    foreach label : all_labels
        if label in document.labels
            document.labels_vector.push(+1)
        else
            document.labels_vector.push(-1)

Questa procedura in teoria si può eseguire una volta sola in assoluto: se l'output si potesse salvare su un file non sarebbe male.

foreach document : all_documents
    file.write(document.mega_vector)
    file.write(document.labels_vector)

'''
