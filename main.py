import parserFile
from miml_svm import MiMlSVM
import random

if __name__ == "__main__":
    # Open the first Reuters data set and create the parser
    # filename = "dataset/reut2-000.sgm"
    # parser = parserFile.ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    # doc = parser.parse(open(filename, 'rb'))
    # print list(doc)[0][0]
    # pprint.pprint(list(doc))
    datasetSize = 100
    testsetSize = 20
    featuresNumber = 2
    labelsNumber = 3
    minInstances = 5
    maxInstances = 20
    dataset = []
    for document in range(datasetSize):
        instances = []
        for instance in range(random.randint(minInstances, maxInstances + 1)):
            features = []
            for feature in range(0, featuresNumber):
                features.append(random.random())
            instances.append(features)
        dataset.append(instances)

    labels = []
    for document in range(datasetSize):
        docLabels = []
        for label in range(labelsNumber):
            if(random.random() < 0.5):
                docLabels.append(-1)
            else:
                docLabels.append(+1)
        labels.append(docLabels)


    miml = MiMlSVM()
    miml.train(dataset, labels)

    testset = []
    for document in range(datasetSize):
        instances = []
        for instance in range(random.randint(minInstances, maxInstances + 1)):
            features = []
            for feature in range(0, featuresNumber):
                features.append(random.random())
            instances.append(features)
        testset.append(instances)

    print miml.test(testset)
