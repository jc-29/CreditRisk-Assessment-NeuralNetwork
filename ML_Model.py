from tensorflow import keras
import numpy as np

class Model():
    def __init__(self):
        '''
        To use the model:
        1. Call the init_model() function with the number of inputs, number of outputs and number of hidden neurons
        2. Call the train() function with the number of epochs and batch size
        3. Call the evaluate() function with the batch size (only if you want to evaluate the model's performance using the test data)
        4. Call the predict() function with a 1D numpy array of len=11, where each element is a feature value
        '''
        self.file = open('ml_model/data/credit_risk_dataset_preprocessed_median.csv', 'r')
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.train_labels = None
        self.validation_labels = None
        self.test_labels = None
        self.model = None
    
    def get_data(self):
        '''
        This function reads the data from the file and splits it into training, validation and testing sets. 
        All the data is stored in numpy arrays.
        '''
        self.data = np.genfromtxt(self.file, delimiter=',', encoding=None)

        # split the data into training, validation and testing sets
        self.train_data = np.array(self.data[1:int(len(self.data) * 0.6)]) # 60% of the data for training
        self.validation_data = np.array(self.data[int(len(self.data) * 0.6) + 1 : int(len(self.data) * 0.8):]) # 20% of the data for validation
        self.test_data = np.array(self.data[int(len(self.data) * 0.8):]) # 20% of the data for testing

        # separate the features from the labels
        self.train_labels = self.train_data[:, 8]
        self.validation_labels = self.validation_data[:, 8]
        self.test_labels = self.test_data[:, 8]

        # remove the labels from the data
        self.train_data = np.delete(self.train_data, 8, 1)
        self.validation_data = np.delete(self.validation_data, 8, 1)
        self.test_data = np.delete(self.test_data, 8, 1)
        
        self.file.close()
    
    def init_model(self, num_inputs, num_outputs, num_hidden_neurons):
        '''
        This model assumes only one hidden layer with num_hidden_neurons neurons. 
        Each of the 3 function parameters can be changed for testing and improving purposes. 
        The optimizer and loss functions can also be changed for testing and improving purposes.
        '''
        self.model = keras.models.Sequential([keras.layers.InputLayer(input_shape=(num_inputs,)),
            keras.layers.Dense(num_hidden_neurons, activation="sigmoid"),
            keras.layers.Dense(num_outputs, activation="sigmoid")
            ])
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def train(self, epochs, batch_size):
        '''
        The number of epochs and batch size can be changed for testing and improving purposes.
        '''
        self.get_data()
        self.model.fit(self.train_data, self.train_labels, epochs=epochs, batch_size=batch_size)

    def evaluate(self, batch_size):
        '''
        The batch size can be changed for testing and improving purposes.
        '''
        self.model.evaluate(self.test_data, self.test_labels, batch_size=batch_size)

    def predict(self, data):
        '''
        data is a 2D numpy array where each sub-array is of len=11 and represents the features of an individual. 
        Returns a 2D numpy array that has the same length as the input data. Each sub-array is of len=2 and represents the probability of the individual defaulting or not defaulting (0 is non-default, 1 is defualt).
        '''
        return self.model.predict(data)

    def save_model(self, filename):
        '''
        Saves the model to a file.
        '''
        self.model.save(filename)

# if __name__ == '__main__':
#     model = Model()
#     model.init_model(11, 1, 8) # (num_inputs, num_outputs, num_hidden_neurons)
#     model.train(30, 15) # (epochs, batch_size)
#     # model.evaluate(15) # (batch_size)
#     # print('=============================================================')
#     # data = np.array([[22, 85000, 1, 6, 3, 1, 35000, 10.37, 0.41, 1, 4], 
#     #                     [21, 10000, 2, 2, 0, 0, 4500, 8.63, 0.45, 1, 2],
#     #                     [23, 95000, 1, 2, 3, 0, 35000, 7.9, 0.37, 1, 2],
#     #                     [26, 108160, 1, 4, 5, 4, 35000, 18.39, 0.32, 1, 4],
#     #                     [23, 115000, 1, 2, 5, 0, 35000, 7.9, 0.3, 1, 4],
#     #                     [23, 500000, 0, 7, 4, 1, 30000, 10.65, 0.06, 1, 3],
#     #                     [23, 120000, 1, 0, 5, 0, 35000, 7.9, 0.29, 1, 4]]) # 1, 1, 1, 1, 0, 0, 0

#     # print('Model Prediction: \n', model.predict(data))  
#     # print('=============================================================')
#     model.save_model('saved_model.h5')