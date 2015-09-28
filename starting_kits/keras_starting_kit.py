from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):

    def __init__(self):
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, 3, border_mode='full'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2, 2)))

        model.add(Convolution2D(128, 3, 3, 3, border_mode='full'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2, 2)))

        model.add(Convolution2D(128, 3, 3, 3, border_mode='full'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64*8*8, 256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, 10))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        self.model = model
        self.sgd = sgd

    def fit(self, X, y):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd)
        self.pipeline = make_pipeline(StandardScaler(), self.model)
        self.pipeline.fit(X, y)

    def predict(self, X):
        self.pipeline.predict(X)

    def predict_proba(self, X):
        self.pipeline.predict_proba(X)
