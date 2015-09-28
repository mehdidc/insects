import numpy as np
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, average_precision_score

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "classifier"
    module = __import__(name)

    data = np.load("train_64x64.npz")
    X_array = data['X']
    y_array = data['y']
    test_size = 0.2
    random_state = 15
    nb_iter = 1
    cv = StratifiedShuffleSplit(y_array, nb_iter,
                                test_size=test_size,
                                random_state=random_state)
    for train_is, test_is in cv:
        clf = module.Classifier()
        clf.fit(X_array[train_is], y_array[train_is])
        score = average_precision_score(clf, X_array[test_is],
                                        y_array[test_is])
        print("Average precision : {0}".format(score))
