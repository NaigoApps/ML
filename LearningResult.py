import numpy as np
import sklearn.metrics

class Label:
    def __init__(self, id, rank, predicted):
        self._id = id
        self._rank = rank
        self._predicted = predicted

    def getId(self):
        return self._id

    def getRank(self):
        return self._rank

    def getPredicted(self):
        return self._predicted

    def __cmp__(self, other):
        if self._rank < other._rank:
            return -1
        if self._rank == other._rank:
            return 0
        return 1

    def __eq__(self, other):
        return self._id == other._id

    def __hash__(self):
        return self._id

class Labels:
    def __init__(self, predictions, actuals):
        self._label_count = len(actuals)
        self._predicted_labels = set()
        self._not_predicted_labels = set()
        self._actual_labels = set()
        self._not_actual_labels = set()

        computed_indexes = set()
        rank = 1
        while len(computed_indexes) < len(predictions):
            best_i = None
            # Find best prediction
            for (i, prediction) in enumerate(predictions):
                if i not in computed_indexes and (best_i is None or prediction > predictions[best_i]):
                    best_i = i
            if best_i is not None:
                if predictions[best_i] > 0:
                    self._predicted_labels.add(Label(best_i, rank, predictions[best_i]))
                else:
                    self._not_predicted_labels.add(Label(best_i, rank, predictions[best_i]))
                rank += 1
                computed_indexes.add(best_i)

        for (i, actual) in enumerate(actuals):
            if actual > 0:
                self._actual_labels.add(Label(i, 0, actual))
            else:
                self._not_actual_labels.add(Label(i, 0, actual))

    def symmetric_difference(self):
        return len(self._predicted_labels.symmetric_difference(self._actual_labels))

    def best_predicted(self):
        if len(self._predicted_labels) > 0:
            return next(label for label in self._predicted_labels if label.getRank() == 1)
        return None

    def best(self):
        if len(self._predicted_labels) > 0:
            return next(label for label in self._predicted_labels if label.getRank() == 1)
        else:
            return next(label for label in self._not_predicted_labels if label.getRank() == 1)
        return None

    def sorted_predictions(self):
        all_predictions = self._not_predicted_labels.union(self._predicted_labels)
        return sorted(all_predictions)

    def findLabel(self, id):
        for label in self._predicted_labels:
            if label.getId() == id:
                return label
        for label in self._not_predicted_labels:
            if label.getId() == id:
                return label
        return None

    def actual_count(self):
        return len(self._actual_labels)

    def predicted_count(self):
        return len(self._predicted_labels)

    def labels_count(self):
        return self._label_count

class LearningResult:
    def __init__(self, predictions, actuals):
        self._labels = []
        for prediction, actual in zip(predictions, actuals):
            self._labels.append(Labels(prediction, actual))

    def hamming_loss(self):
        sum = 0.0
        for labels in self._labels:
            sum += float(labels.symmetric_difference()) / labels.labels_count()
        return sum / len(self._labels)

    def one_error(self):
        sum = 0.0
        for labels in self._labels:
            label = labels.best_predicted()
            if label is not None and label not in labels._actual_labels:
                sum += 1
        return sum / len(self._labels)

    def coverage(self):
        sum = 0.0
        for labels in self._labels:
            # Posso sbagliare solo se ci sono label da predirre
            if len(labels._actual_labels) > 0:
                sorted_labels = labels.sorted_predictions()
                rank = -1
                # Tra tutti i label prendo quello predetto meglio (anche se non e' predetto correttamente)
                for label in sorted_labels:
                    if rank == -1 and label in labels._actual_labels:
                        rank = label.getRank()
                # Sommo il rank del label migliore:
                # se il miglior label (anche se predetto negativo) e' un label effettivo sono contento
                sum += rank - 1
        return sum / len(self._labels)

    def ranking_loss(self):
        sum = 0.0
        for labels in self._labels:
            cur_sum = 0.0
            for neg_label in labels._not_actual_labels:
                for pos_label in labels._actual_labels:
                    if labels.findLabel(neg_label.getId()).getRank() < labels.findLabel(pos_label.getId()).getRank():
                        cur_sum += 1
            if cur_sum > 0:
                cur_sum /= len(labels._not_actual_labels) * len(labels._actual_labels)
                sum += cur_sum
        return sum / len(self._labels)

    def average_precision(self):
        sum = 0.0
        for labels in self._labels:
            if len(labels._actual_labels) > 0:
                cur_sum = 0.0
                for pos_label in labels._actual_labels:
                    for pos_label_1 in labels._actual_labels:
                        if labels.findLabel(pos_label_1.getId()).getRank() <= labels.findLabel(pos_label.getId()).getRank():
                            cur_sum += 1
                    cur_sum /= labels.findLabel(pos_label.getId()).getRank()
                cur_sum /= len(labels._actual_labels)
                sum += cur_sum
            else:
                sum += 1
        return sum / len(self._labels)

    def average_recall(self):
        sum = 0.0
        for labels in self._labels:
            if len(labels._actual_labels) > 0:
                cur_sum = 0.0
                for label in labels._actual_labels:
                    if labels.findLabel(label.getId()).getPredicted() > 0:
                        cur_sum += 1
                cur_sum /= len(labels._actual_labels)
                sum += cur_sum
            else:
                sum += 1
        return sum / len(self._labels)

    def average_F1(self):
        prec = self.average_precision()
        rec = self.average_recall()
        return 2*prec*rec/(prec + rec)