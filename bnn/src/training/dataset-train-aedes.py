import numpy as np
import os
import random
import sys
import os
import time

from argparse import ArgumentParser

import theano
import theano.tensor as T
import lasagne

import cPickle as pickle

import quantized_net
import cnv

from collections import OrderedDict

class loadDataset():
    def __init__(self,path):
        self.dir= path

    def load_images(self,filename):
        with open(os.path.join(self.dir,filename),'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=0)

        data=data.reshape(-1,3,32,32)
	data=data.transpose([0,1,3,2])
        return data

    def load_labels(self,filename):
        with open(os.path.join(self.dir,filename),'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=0)
        return labels

def main(args):
    learning_parameters = OrderedDict()
    #Quantization parameters
    learning_parameters.activation_bits = args.activation_bits
    print("activation_bits = " + str(learning_parameters.activation_bits))
    learning_parameters.weight_bits = args.weight_bits
    print("weight_bits = " + str(learning_parameters.weight_bits))
    # BN parameters
    batch_size = 20
    print("batch_size = " + str(batch_size))
    # alpha is the exponential moving average factor
    learning_parameters.alpha = .1
    print("alpha = " + str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = " + str(learning_parameters.epsilon))
    # W_LR_scale = 1.
    learning_parameters.W_LR_scale = "Glorot"  # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = " + str(learning_parameters.W_LR_scale))
    # Training parameters
    num_epochs =1000
    print("num_epochs = " + str(num_epochs))
    # Decaying LR
    LR_start = 0.001
    print("LR_start = " + str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = " + str(LR_fin))
    LR_decay = (LR_fin / LR_start) ** (1. / num_epochs)
    print("LR_decay = " + str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    save_path = "dataset-%dw-%da.npz" % (learning_parameters.weight_bits, learning_parameters.activation_bits)
    #save_path = "dataset-parameters.npz"
    print("save_path = " + str(save_path))
    train_set_size = 870
    print("train_set_size = " + str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = " + str(shuffle_parts))
    print('Loading the dataset...')
    dataset = loadDataset('data')
    train_set_X = dataset.load_images(filename='train-images.bin')
    train_set_y = dataset.load_labels(filename='train-labels.bin')
    test_set_X = dataset.load_images(filename='test-images.bin')
    test_set_y = dataset.load_labels(filename='test-labels.bin')

    list = []
    for i in range(train_set_X.shape[0]):
        list.append({'X': train_set_X[i], 'y': train_set_y[i]})
    random.shuffle(list)
    random.shuffle(list)
    random.shuffle(list)
    train_set_X = np.zeros([train_set_X.shape[0], 3, 32, 32], dtype=np.uint8)
    train_set_y = np.zeros([train_set_y.shape[0]], dtype=np.uint8)
    for i in range(train_set_X.shape[0]):
        train_set_X[i] = np.expand_dims(list[i]['X'], axis=0)
        train_set_y[i] = np.expand_dims(list[i]['y'], axis=0)
    train_set_X = np.subtract(np.multiply(2. / 255., train_set_X), 1., dtype=np.float32)
    test_set_X = np.subtract(np.multiply(2. / 255., test_set_X), 1., dtype=np.float32)
    #valid_set_X = train_set_X[(train_set_size):(train_set_size + valid_set_size)]
    #valid_set_y = train_set_y[(train_set_size):(train_set_size + valid_set_size)]
    #train_set_X = train_set_X[:(train_set_size)]
    #train_set_y = train_set_y[:(train_set_size)]
    valid_set_X=test_set_X
    valid_set_y=test_set_y

    # flatten targets
    train_set_y = np.hstack(train_set_y)
    valid_set_y = np.hstack(valid_set_y)
    test_set_y = np.hstack(test_set_y)

    # Onehot the targets
    train_set_y = np.float32(np.eye(5)[train_set_y])
    valid_set_y = np.float32(np.eye(5)[valid_set_y])
    test_set_y = np.float32(np.eye(5)[test_set_y])

    # for hinge loss
    train_set_y = 2 * train_set_y - 1.
    valid_set_y = 2 * valid_set_y - 1.
    test_set_y = 2 * test_set_y - 1.

    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)


    cnn = cnv.genCnv(input, 5, learning_parameters)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    # W updates
    W = lasagne.layers.get_all_params(cnn, quantized=True)
    W_grads = quantized_net.compute_grads(loss, cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = quantized_net.clipping_scaling(updates, cnn)

    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, quantized=False)
    updates = OrderedDict(
        updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')

    quantized_net.train(
        train_fn, val_fn,
        cnn,
        batch_size,
        LR_start, LR_decay,
        num_epochs,
        train_set_X, train_set_y,
        valid_set_X, valid_set_y,
        test_set_X, test_set_y,
        save_path=save_path,
        shuffle_parts=shuffle_parts)

    """""
    image=train_set_X[823]
    image=np.transpose(image,[2,1,0])

    from PIL import Image

    image=Image.fromarray(image)

    image.save('test.png',format='png')
    """

if __name__ == "__main__":
    # Parse some command line options
    parser = ArgumentParser(
        description="Train the LFC network on the own dataset")
    parser.add_argument('-ab', '--activation-bits', type=int, default=2, choices=[1,2],
        help="Quantized the activations to the specified number of bits, default: %(default)s")
    parser.add_argument('-wb', '--weight-bits', type=int, default=2, choices=[1,2],
        help="Quantized the weights to the specified number of bits, default: %(default)s")
    args = parser.parse_args()
    main(args)
    
