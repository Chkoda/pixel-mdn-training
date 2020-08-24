""" eval_nn: module to evaluate a dataset using a neural network """

import argparse

from keras.callbacks import ModelCheckpoint
#from keras.layers.core import Dense, Activation
#from keras.models import Sequential
#from keras.optimizers import SGD
#import keras.regularizers
import numpy as np

from keras_utils import Sigmoid, Profile, ThresholdEarlyStopping
import root_utils
import utils

import theano
import matplotlib.pyplot as plt
from theano import tensor as T, function, printing


np.random.seed(12122)
import keras
import keras.backend as K
import h5py
from keras.utils import plot_model

from scipy.integrate import quad

import itertools as it
import sqlite3

import keras.models
import numpy as np




__all__ = ['eval_nn']


def eval_nn(inputp,
            config,
            output,
            normalization,
            network_type,
            model_file): \
            # pylint: disable=too-many-arguments
    """ evaluate a dataset  with a neural network stored on disk

    arguments:
    inputp -- path to the ROOT dataset
    model -- path to the yaml keras model config file
    config -- path to the branches config file
    output -- output name for the sqlite database (overwrites the 'test' table)
    normalization -- path to the txt file with normalization constants
    """

    model = keras.models.load_model(
        #'Outputs/Try_3p/MDN_Three1Gauss_MC16_v3-150ep.h5', #MDN_Two1Gauss_MC16_v6-250ep.h5,  'Outputs/Try_2DGauss/pos1_MC16.h5'
        #'/home/elham/NN_optimise/pixel-MDN-training/Outputs/Try_2DGauss/pos1_MC16.h5',
        model_file,
        custom_objects={
            'loss': mixture_density_loss(nb_components=1)
        }
    )

    _eval_dataset(
        model=model,
        network_type=network_type,
        path=inputp,
        tree='NNinput',
        branches=utils.get_data_config_names(config, meta=True),
        norm=utils.load_normalization(normalization),
        dbpath=output
    )


def mixture_density_loss(nb_components, target_dimension=2):

        """ Compute the mixture density loss:
            \begin{eqnarray}
            P(Y|X) = \sum_i P(C_i) N(Y|mu_i(X), beta_i(X)) \\
            Loss(Y|X) = - log(P(Y|X))
            \end{eqnarray}
        """

        def loss(y_true, y_pred):

            batch_size = K.shape(y_pred)[0]

            # Each row of y_pred is composed of (in order):
            # 'nb_components' prior probabilities
            # 'nb_components'*'target_dimension' means
            # 'nb_components'*'target_dimension' precisions
            priors = y_pred[:,:nb_components]

            m_i0 = nb_components
            m_i1 = m_i0 + nb_components * target_dimension
            means = y_pred[:,m_i0:m_i1]
            #means = theano.printing.Print('means')(means)

            #y_true = theano.printing.Print('true means')(y_true)

            p_i0 = m_i1
            p_i1 = p_i0 + nb_components * target_dimension
            precs = y_pred[:,p_i0:p_i1]
            #precs = theano.printing.Print('precs')(precs)

            # Now, compute the (x - mu) vector. Have to reshape y_true and
            # means such that the subtraction can be broadcasted over
            # 'nb_components'
            means = K.reshape(means, (batch_size , nb_components, target_dimension))
            x = K.reshape(y_true, (batch_size, 1, target_dimension)) - means
            #x = T.switch(x >= 1.0 , 1.0, x)
            
            x = theano.printing.Print('x')(x)


            # Compute the dot-product over the target dimensions. There is
            # one dot-product per component per example so reshape the
            # vectors such that a batch_dot product can be carried over
            # the axis of target_dimension
            x = K.reshape(x, (batch_size * nb_components, target_dimension))
            precs = K.reshape(precs, (batch_size * nb_components, target_dimension))
            #precs = theano.printing.Print('precs')(precs)
            #std = K.reshape(precs, (batch_size * nb_components, target_dimension))
            #invStdsq = 1/(std*std)

            # reshape the result into the natural structure
            expargs = K.reshape(K.batch_dot(-0.5 * x * precs, x, axes=1), (batch_size, nb_components))
            #expargs = np.max(expargs,5.0)
            #expargs = K.reshape(K.batch_dot(-0.5 * x * invStdsq, x, axes=1), (batch_size, nb_components))
            #expargs = theano.printing.Print('expargs')(expargs)

            # There is also one determinant per component per example
            dets = K.reshape(K.abs(K.prod(precs, axis=1)), (batch_size, nb_components))
            #dets = K.reshape(K.abs(K.prod(precs, axis=1)), (batch_size, nb_components))
            #dets = theano.printing.Print('dets')(dets)


            norms = K.sqrt(dets/np.power(2*np.pi,target_dimension)) * priors #/np.power(2*np.pi,target_dimension)
            #norms = theano.printing.Print('norms')(norms)


            # LogSumExp, for enhanced numerical stability
            x_star = K.max(expargs, axis=1, keepdims=True)
            #x_star = theano.printing.Print('x_star')(x_star)
            logprob =  -x_star - K.reshape(K.log(K.sum(norms * K.exp(expargs - x_star), axis=1)),(-1, 1))
            #logprob =  -x_star -  K.log(K.sum(norms * K.exp(expargs - x_star), axis=1))
            #logprob = theano.printing.Print('logprob')(logprob)

            #logprob = - K.log(norms*K.exp(expargs))
            #logprob = theano.printing.Print('logprob')(logprob)

            logprob = T.switch(logprob >= 10.0 , 0.0, logprob)
            return logprob

        return loss



def _eval_dataset(model,
                  network_type,
                  path,
                  tree,
                  branches,
                  norm,
                  dbpath,
                  batch=128): \
                  # pylint: disable=too-many-arguments

    dbconn = sqlite3.connect(dbpath)
    _prepare_db(
        dbconn=dbconn,
        meta_branches=branches[2],
        y_branches=branches[1]
    )

    data_generator = root_utils.generator(
        path,
        tree=tree,
        branches=branches[:2],
        batch=batch,
        normalization=norm,
        loop=False
    )

    meta_generator = root_utils.generator(
        path,
        tree=tree,
        branches=(branches[2], []),
        batch=batch,
        normalization=None,
        loop=False
    )

    nb_components = 1


    for (xbatch, ybatch), (meta, _) in zip(data_generator, meta_generator):
        y_pred=model.predict(xbatch, batch_size=xbatch.shape[0])
        if network_type == '1particle':
            pred_mean = y_pred[:,1:3]
            pred_prec = y_pred[:,3:5]
        elif network_type == '2particle':
            pred_mean = np.hstack((y_pred[0][:,1:3], y_pred[1][:,1:3]))
            pred_prec = np.hstack((y_pred[0][:,3:5], y_pred[1][:,3:5]))
        elif network_type == '3particle':
            pred_mean = np.hstack((y_pred[0][:,1:3], y_pred[1][:,1:3], y_pred[2][:,1:3]))
            pred_prec = np.hstack((y_pred[0][:,3:5], y_pred[1][:,3:5], y_pred[2][:,3:5]))

        _insert_into_db(
            dbconn,
            meta,
            y_truth=ybatch,
            y_pred=pred_mean,
            y_uncert=1/np.sqrt(pred_prec)
        )

    dbconn.commit()


def _insert_into_db(dbconn, meta, y_truth, y_pred, y_uncert):

    mcol = meta.shape[1]
    ycol = y_truth.shape[1]

    shape = (meta.shape[0], mcol + 3*ycol)

    meta_begin = 0
    meta_end = meta_begin + mcol
    truth_begin = meta_end
    truth_end = truth_begin + ycol
    pred_begin = truth_end
    pred_end = pred_begin + ycol
    uncert_begin = pred_end
    uncert_end = uncert_begin + ycol

    data = np.empty(shape)
    data[:, meta_begin:meta_end] = meta
    data[:, truth_begin:truth_end] = y_truth
    data[:, pred_begin:pred_end] = y_pred
    data[:, uncert_begin:uncert_end] = y_uncert

    sql = 'INSERT INTO test VALUES ('
    sql += ','.join(['?'] * shape[1])
    sql += ')'

    dbconn.executemany(sql, data)


def _prepare_db(dbconn, meta_branches, y_branches):

    dbconn.execute('DROP TABLE IF EXISTS test')
    sql = 'CREATE TABLE test ('
    sql += ','.join([br + ' REAL' for br in meta_branches])
    sql += ','
    sql += ','.join([br + '_TRUTH REAL' for br in y_branches])
    sql += ','
    sql += ','.join([br + '_PRED REAL' for br in y_branches])
    sql += ','
    sql += ','.join([br + '_PRED_UNCERT REAL' for br in y_branches])
    sql += ')'
    dbconn.execute(sql)
    dbconn.commit()


def _main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", required=True,
        help='input root file name with full path')
    parse.add_argument("--config", required=True,
        help='network config option. example: <(python $PWD/genconfig.py --type pos1)')
    parse.add_argument("--output", required=True,
        help='output .db file name with full path')
    parse.add_argument("--normalization", required=True,
        help='normalization text file name')
    parse.add_argument('--network_type', required=True,
        help='Network type: 1particle, 2particle, 3particle')
    parse.add_argument("--model_file", required=True,
        help='model file name (.h5) with path')
    args = parse.parse_args()

    #Exponential loss function
    keras.activations.abs = K.abs


    eval_nn(
        args.input,
        args.config,
        args.output,
        args.normalization,
        args.network_type,
        str(args.model_file)
    )


if __name__ == '__main__':
    _main()
