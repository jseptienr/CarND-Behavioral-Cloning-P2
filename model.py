import csv
import cv2
import sklearn
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D


def load_data(csv_path):
    '''Loads data from specified path to csv. '''
    samples = []
    print('Retreiving data from {}'.format(csv_path))
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    del samples[0]
    return samples


def generator(samples, batch_size=32, training=True):
    '''Generates batches from given samples. '''
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # If training hoose camera sample randomly
                # center, left, right => 0, 1, 2
                if training:
                    camera_source = np.random.randint(0, 3)
                else:
                    camera_source = 0
                name = 'data/IMG/'+batch_sample[camera_source].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[3])
                # Make angle correction if not center camera
                if camera_source == 1:
                    angle = angle + 0.2
                if camera_source == 2:
                    angle = angle - 0.2
                # Flip image randomly
                if training  and np.random.rand() > 0.5:
                    image = np.fliplr(image)
                    angle = -angle
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    '''Creates  convolutional model from NVIDIA. '''
    model = Sequential()
    # Preprocess samples
    # Normalize image between (-1, 1)
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
    # Crop background and hood of car output 65x320x3
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    # Convolutional layer 5x5 output: 31x158x24
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    # Convolutional layer 5x5 output: 14x77x36
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    # Convolutional layer 5x5 output: 5x37x48
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    # Convolutional layer 3x3 output: 3x35x64
    model.add(Convolution2D(64,3,3, activation="relu"))
    # Convolutional layer 5x5 output: 1x33x64
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    # Fully connected layer with dropout ouput: 100x50
    model.add(Dense(100))
    model.add(Dropout(0.5))
    # Fully connected layer with dropout ouput: 50x10
    model.add(Dense(50))
    model.add(Dropout(0.5))
    # Fully connected layer output: 10x1
    model.add(Dense(10))
    # Output 1
    model.add(Dense(1))
    return model


def train_model(samples, model):
    '''Trains the model using mean square error and Adam Optimizer'''
    # Shuffle samples and split training and validation sets
    samples = shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('Number of training samples: {}'.format(len(train_samples)))
    print('Number of validation samples: {}'.format(len(validation_samples)))

    # Training and validation generators
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples, training=False)

    # Use mean squared error and Adam optimizer
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=32000,
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=5)
    return history_object


def show_loss(history_object):
    '''Plots the training and validation loss for each epoch. '''
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def save_model(filename, model):
    '''Saves the model and weights to h5 and json'''
    model.save('{}.h5'.format(filename))
    model.save_weights('{}_weights.h5'.format(filename))
    with open('{}.json'.format(filename), 'w') as modelfile:
        json.dump(model.to_json(), modelfile)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Invalid number of arguments!')
        print('Usage: python model.py [path to csv] [model filename]')
        sys.exit(0)
    data_path = sys.argv[1]
    model_filename = sys.argv[2]

    # Pipeline
    print('Loading data from {}'.format(data_path))
    samples = load_data(data_path)
    print('Creating model...')
    model = create_model()
    print('Training model...')
    history_object = train_model(samples, model)
    #show_loss(history_object)
    print('Saving model...')
    save_model(model_filename, model)
