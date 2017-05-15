import misvm
import numpy as np
import time
import gc

class MiMlSVM:
    def __init__(self):
        self.SVMs = None;

    def train(self, dataset, labels):
        self.SVMs = []
        index = 0
        glob_start = time.time()
        for label in np.transpose(labels):
            # classifier = misvm.MISVM(kernel='linear', C=1.0, max_iters=50)
            start = time.time()
            classifier = misvm.SIL(kernel='linear', C=1.0)
            classifier.fit(dataset, np.array(label))
            self.SVMs.append(classifier)
            index += 1
            print "Trained " + str(index) + " of " + str(len(np.transpose(labels))) + " in " + str(int(time.time()) - int(start)) + " sec"
        print "It took " + str((int(time.time()) - int(glob_start))/60) + " minutes"

    def test(self, test_set):
        all_labels = []
        index = 0
        for SVM in self.SVMs:
            all_labels.append(SVM.predict(test_set))
            index += 1
            print "Tested " + str(index) + " labels of " + str(len(self.SVMs))
        return np.sign(np.transpose(all_labels))
