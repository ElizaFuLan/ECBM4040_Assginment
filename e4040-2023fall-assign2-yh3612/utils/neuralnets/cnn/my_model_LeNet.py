#!/usr/bin/env/ python
# ECBM E4040 Fall 2023 Assignment 2

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model
import time

class LeNet(Model):
    """
    LeNet is an early and famous CNN architecture for image classfication task.
    It is proposed by Yann LeCun. Here we use its architecture as the startpoint
    for your CNN practice. Its architecture is as follow.

    input >> Conv2DLayer >> Conv2DLayer >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    Or

    input >> [conv2d-avgpooling] >> [conv2d-avgpooling] >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, input_shape, output_size=10):
        '''
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(LeNet, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape, padding="same")
        self.avgpool1 = tf.keras.layers.AveragePooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same")
        self.avgpool2 = tf.keras.layers.AveragePooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(output_size, activation='softmax')
        #############################################################
        #                          END TODO                         #                                              
        #############################################################

    
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        return self.fc1(x)
        #############################################################
        #                          END TODO                         #                                              
        #############################################################

def my_training_task5(X_train, y_train, X_val, y_val, conv_featmap=[32, 32, 32], fc_units=[84, 84],
                      conv_kernel_size=[5, 5, 5], pooling_size=[2, 2, 2], l2_norm=0.005, seed=114514,
                      learning_rate=0.005, epoch=20, batch_size=245, verbose=False, pre_trained_model=None):

    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # Assuming LeNet function returns a tf.keras model
    model = LeNet(input_shape=(128, 128, 3), output_size=5)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Create tf.data datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Load pre-trained model if given
    if pre_trained_model:
        model.load_weights(pre_trained_model)

    best_acc = 0

    # Training loop
    for epc in range(epoch):
        for x_batch, y_batch in train_dataset:
            history = model.train_on_batch(x_batch, y_batch)
            if verbose:
                print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epc, history['loss'], history['accuracy']))

        # Evaluate on validation data
        val_loss, val_accuracy = model.evaluate(val_dataset)
        if verbose:
            print('Validation loss: {}, Validation accuracy : {}%'.format(val_loss, val_accuracy * 100))
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            
    print("Training ends. Best valid accuracy is {}.".format(best_acc * 100))
    return model
