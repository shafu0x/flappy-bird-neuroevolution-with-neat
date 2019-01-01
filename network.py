from keras.models import Sequential
from keras.layers import Dense

import numpy as np

MARGIN = 0.025
THRESHOLD = 0.5

class NN:
    def __init__(self):
        self.output = []
        self.model = Sequential()
        self.model.add(Dense(units=16, activation='sigmoid', input_dim=3))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def get_average_output(self):
        return sum(self.output)/len(self.output)

    def is_average_prediction_good(self, threshold, margin=0.05):
        average_pred = self.get_average_output()
        if average_pred < threshold + MARGIN and average_pred > threshold - MARGIN:
            print('average_pred {}'.format(average_pred))
            return True
        else:
            return False

    def predict(self, input):
        prediction = self.model.predict(input)[0][0]
        self.output.append(prediction)
        if prediction > THRESHOLD:
            return 1
        else:
            return 0
