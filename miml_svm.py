import misvm
import numpy

class MiMlSVM:
    def __init__(self):
        self.SVMs = None;

    def train(self, dataset, labels):
        self.SVMs = []
        index = 0
        for label in numpy.transpose(labels):
            # classifier = misvm.MISVM(kernel='linear', C=1.0, max_iters=50)
            classifier = misvm.SIL(kernel='linear', C=1.0)
            classifier.fit(dataset, label)
            self.SVMs.append(classifier)
            index += 1
            print "Trained " + str(index) + " of " + str(len(numpy.transpose(labels)))

    def test(self, test_set):
        all_labels = []
        index = 0
        for SVM in self.SVMs:
            all_labels.append(SVM.predict(test_set))
            index += 1
            print "Tested " + str(index) + " labels of " + str(len(self.SVMs))
        return numpy.transpose(all_labels)
