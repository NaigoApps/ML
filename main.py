import parserFile
from miml_svm import MiMlSVM
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class Pixel:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class BigRect:
    def __init__(self, x, y, w, h, parts):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rects = []
        for r in range(parts):
            for c in range(parts):
                self.rects.append(Rect(x + float(c) / parts * w,
                                       y + float(r) / parts * h,
                                       w / parts,
                                       h / parts))


def generateRect(rows, cols, minS, maxS, parts):
    x = int(random.random() * cols)
    y = int(random.random() * rows)
    w = int(min(cols - x, random.random() * (maxS - minS) + minS))
    h = int(min(rows - y, random.random() * (maxS - minS) + minS))
    return BigRect(x, y, w, h, parts)


def randomRect(rows, cols):
    x = int(random.random() * cols)
    y = int(random.random() * rows)
    w = int(random.random() * (cols - x))
    h = int(random.random() * (rows - y))
    return Rect(x, y, w, h)


def stretch(l, r, x):
    x = (x - l) / (r - l)
    x = max(x, l)
    x = min(x, r)
    return x


def calcLabel(rect, img):
    colors = [-1, -1, -1]
    for r in range(int(rect.y), int(rect.y + rect.h)):
        for c in range(int(rect.x), int(rect.x + rect.w)):
            if img[r][c][0] == 1:
                colors[0] = 1
            if img[r][c][1] == 1:
                colors[1] = 1
            if img[r][c][2] == 1:
                colors[2] = 1
    return colors

def convert(img, rects):
    dataset = []
    labels = []
    for bigRect in rects:
        document = []
        for littleRect in bigRect.rects:
            document.append([littleRect.x, littleRect.y, littleRect.w, littleRect.h])
        dataset.append(document)
        labels.append(calcLabel(bigRect, img))
    return (dataset, labels)


if __name__ == "__main__":

    nRows = 10
    nCols = 10
    steps = 1

    minRectSize = 5
    maxRectSize = 20

    datasetSize = 200
    trainingSetSize = 199
    testSetSize = 1

    image = np.zeros((nRows, nCols, 3))
    for s in range(steps):
        rect = randomRect(nRows, nCols)
        image[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w] = [1, 0, 0]
        rect = randomRect(nRows, nCols)
        image[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w] = [0, 1, 0]
        rect = randomRect(nRows, nCols)
        image[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w] = [0, 0, 1]

    _ , p = plt.subplots(1, 1)
    p.imshow(image)

    rawDataset = []

    for i in range(datasetSize):
        rawDataset.append(generateRect(nRows, nCols, minRectSize, maxRectSize, int(random.random() * 2 + 2)))

    (dataset, labels) = convert(image, rawDataset)

    for i, r in enumerate(rawDataset):
        # for sr in r.rects:
            # p[1].add_patch(Rectangle((sr.x, sr.y), sr.w, sr.h, fill=False, edgecolor="red"))
        p.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, edgecolor=[(l + 1) / 2 * 0.7 for l in labels[i]]))

    plt.xlim((0, nCols))
    plt.ylim((0, nRows))
    plt.show()

    trainingSet = dataset[0: trainingSetSize]
    trainingSetLabels = labels[0: trainingSetSize]
    testSet = dataset[trainingSetSize::]
    testSetLabels = labels[trainingSetSize::]

    svm = MiMlSVM()
    svm.train(trainingSet, trainingSetLabels)
    result = svm.test(testSet)

    comparison = np.sign(testSetLabels) - np.sign(result)

    ok = 0
    perfect = 0
    for entry in comparison:
        if (entry == [0,0,0]).all():
            perfect += 1
        if entry[0] == 0:
            ok += 1
        if entry[1] == 0:
            ok += 1
        if entry[2] == 0:
            ok += 1

    print "Labels ok: " + str(ok) + " of " + str(len(comparison)*3) + " : " + str(ok*100/len(comparison)/3) + "%"
    print "Labels perfect: " + str(perfect) + " of " + str(len(comparison)) + " : " + str(perfect*100/len(comparison)) + "%"