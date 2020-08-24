import os
os.environ['KERAS_BACKEND'] = 'theano'
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

#keras imports
import keras
from keras.layers import Input, Dense, Merge
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.regularizers import l2
import keras.backend as K

#theano import
from theano import tensor as T, function, printing

#local import
import utils
np.random.seed(12122)



def train_nn(training_input,
             validation_fraction,
             output,
             config,
             structure,
             #activation='sigmoid',
             #output_activation='softmax',
             learning_rate,
             regularizer,
             #momentum=0.4,
             #batch=60,
             #min_epochs=10,
             #max_epochs=230,
             ###patience_increase=1.75,
             ###threshold=0.995,
             #min_delta=0.001,
             patience=5,
             #profile=False,
             #nbworkers=6,
             #save_all=False,
             verbose=True): \
            # pylint: disable=too-many-arguments,too-many-locals,dangerous-default-value
    """ train a neural network

    arguments:
    training_input -- path to ROOT training dataset
    validation_fraction -- held-out fraction of dataset for early-stopping
    output -- prefix for output files
    structure -- list with number of hidden nodes in hidden layers
    activation -- non-linearity for hidden layers
    output_activation -- non-linearity of output of the network
    regularizer -- l2 weight regularizer
    momentum -- momentum of weight updates
    batch -- minibatch size
    min_epochs -- minimun number of training epochs
    max_epochs -- maximum number of training epochs
    patience_increase -- amount of patience added when loss improves
    patience -- early stopping tolerance, in number of epochs
    threshold -- threshold to decide if loss improvement is significant
    profile -- create a memory usage log
    nbworkers -- number of parallel thread to load minibatches
    save_all -- save weights at each epoch
    verbose -- bla bla bla level
    """

    #structure = [len(branches[0])] + structure + [len(branches[1])]

#============================= Reading trainign data ============================================================

    with h5py.File('/data/elham/MDN_Study_rel21/'+training_input, 'r') as hf:
        x_train = hf['train_data_x'][:]
        y_train = hf['train_data_y'][:]
        #x_valid = hf['valid_data_x'][:]
        #y_valid = hf['valid_data_y'][:]

    branches = utils.get_data_config_names(config, meta=False)
    print(branches)
#============================= Defining Mixture Density Layer ===================================================

    def mixture_density(nb_components, target_dimension=2):

        """ The Mixture Density output layer. Use with the keras functional api:
            inputs = Inputs(...)
            net = ....
            model = Model(input=[inputs], output=[mixture_density(2)(net)])
        """

        def layer(X):
            pi = Dense(nb_components, activation='softmax')(X)
            mu = Dense(nb_components*target_dimension, activation='linear')(X)
            prec = Dense(nb_components*target_dimension, activation=K.abs)(X)
            return Merge(mode='concat')([pi,mu,prec])

        return layer

#================================== Define the network layers ===================================================

    inputs = Input(shape=(60,))
    print('structure', structure)
    h = Dense(structure[0], activation='relu', W_regularizer=l2(regularizer))(inputs)
    for l in range (0, len(structure)-1):
        h = Dense(structure[l+1], activation='relu', W_regularizer=l2(regularizer))(h)


#================================== Build the model =============================================================

    model = keras.models.Model(inputs=inputs, outputs=[mixture_density(1)(h),mixture_density(1)(h)])

    #print the summary
    print(model.summary())
    plot_model(model, to_file='/home/elham/NN_optimise/pixel-MDN-training/Outputs/Try_2p/'+output+'.png')

    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate, clipnorm=1),
        #optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
        loss=mixture_density_loss(nb_components=1),
    )

    loss_record = LossHistory()
    history = model.fit(
        x=x_train,  #tranning_data[0],
        y=[y_train[:,0:2], y_train[:,2:4]],  #tranning_data[1],
        batch_size=100,
        epochs=230,
        #validation_data=valid_data,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.ModelCheckpoint('/home/elham/NN_optimise/pixel-MDN-training/Outputs/Try_2p/'+output+'.h5', verbose=1, save_best_only=True),
            loss_record
        ],
        verbose=2
    )
    np.savetxt('/home/elham/NN_optimise/pixel-MDN-training/Outputs/Try_2p/training_'+output+'.txt', history.history['loss'])
    np.savetxt('/home/elham/NN_optimise/pixel-MDN-training/Outputs/Try_2p/validation_'+output+'.txt', history.history['val_loss'])


#======================================= prediction history =====================================================
class prediction_history(keras.callbacks.Callback):
      def __init__(self):
         self.predhis = []
         self.ini_pred=[]
      def on_train_begin(self, logs={}):
         self.ini_pred.append(model.predict(x_train))
         #print(model.predict(x_train))  
      def on_epoch_begin(self, epoch, logs={}):
          print("EPOCH BEGIN")
          print(model.predict(x_train))   

      def on_train_end(self, epoch, logs={}):
         self.predhis.append(model.predict(x_train))
         #print("Printing Training data")
         #print(x_train)
         #print("Printing Predictions")\
         #print(model.predict(x_train))

#======================================= Loss history ===========================================================
#Loss after epochs
class LossHistory(keras.callbacks.Callback):
      def on_train_begin(self, logs={}):
        self.losses = []

      def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #print(batch, logs.get('loss'))
        if (np.isnan(logs.get('loss'))):
            print("STOP: NaN")


#================================== Mixture Density Loss ========================================================

def mixture_density_loss(nb_components, target_dimension=2):

    """ Compute the mixture density loss:
        \begin{eqnarray}
        P(Y|X) = \sum_i P(C_i) N(Y|mu_i(X), beta_i(X)) \\
        Loss(Y|X) = - log(P(Y|X))
        \end{eqnarray}
    """

    def loss(y_true, y_pred):

        batch_size = K.shape(y_pred)[0]


        #y_true = y_true[:,0:2]
        #y_true = theano.printing.Print('y_true')(y_true)

        # Each row of y_pred is composed of (in order):
        # 'nb_components' prior probabilities
        # 'nb_components'*'target_dimension' means
        # 'nb_components'*'target_dimension' precisions
        priors = y_pred[:,:nb_components]

        m_i0 = nb_components
        m_i1 = m_i0 + nb_components * target_dimension
        means = y_pred[:,m_i0:m_i1]

        p_i0 = m_i1
        p_i1 = p_i0 + nb_components * target_dimension
        precs = y_pred[:,p_i0:p_i1]

        # Now, compute the (x - mu) vector. Have to reshape y_true and
        # means such that the subtraction can be broadcasted over
        # 'nb_components'
        means = K.reshape(means, (batch_size , nb_components, target_dimension))
        x = K.reshape(y_true, (batch_size, 1, target_dimension)) - means


        # Compute the dot-product over the target dimensions. There is
        # one dot-product per component per example so reshape the
        # vectors such that a batch_dot product can be carried over
        # the axis of target_dimension
        x = K.reshape(x, (batch_size * nb_components, target_dimension))
        precs = K.reshape(precs, (batch_size * nb_components, target_dimension))


        # reshape the result into the natural structure
        expargs = K.reshape(K.batch_dot(-0.5 * x * precs, x, axes=1), (batch_size, nb_components))

        # There is also one determinant per component per example
        dets = K.reshape(K.abs(K.prod(precs, axis=1)), (batch_size, nb_components))

        #precs = T.scalar('precs')
        #dets = T.scalar('dets')
        #c = theano.printing.Print('evaluating precs')(precs) 
        #theano.printing.Print('evaluating dets')(dets) 


        norms = K.sqrt(dets/np.power(2*np.pi,target_dimension)) * priors #/np.power(2*np.pi,target_dimension)

        # LogSumExp, for enhanced numerical stability
        x_star = K.max(expargs, axis=1, keepdims=True)
        logprob =  -x_star - K.reshape(K.log(K.sum(norms * K.exp(expargs - x_star), axis=1)),(batch_size, nb_components))
        logprob = T.switch(logprob >= 10.0 , 0.0, logprob)

        return logprob

    return loss



def _main():

    parse = argparse.ArgumentParser()
    parse.add_argument('--training-input', required=True)
    parse.add_argument('--output', required=True)
    parse.add_argument('--config', required=True)
    parse.add_argument('--validation-fraction', type=float, default=0.1)
    parse.add_argument('--structure', nargs='+', type=int, default=[100, 80, 50])
    parse.add_argument('--learning-rate', type=float, default=0.0001)
    parse.add_argument('--regularizer', type=float, default=0.0001)
    #parse.add_argument('--momentum', type=float, default=0.4)
    #parse.add_argument('--batch', type=int, default=1)
    #parse.add_argument('--min-epochs', type=int, default=10)
    #parse.add_argument('--max-epochs', type=int, default=1000)
    #parse.add_argument('--patience-increase', type=float, default=1.75)
    #parse.add_argument('--threshold', type=float, default=0.995)
    #parse.add_argument('--min-delta', type=float, default=0.0001)
    #parse.add_argument('--patience', type=int, default=5)
    parse.add_argument('--verbose', default=True, action='store_true')
    args = parse.parse_args()

    train_nn(
        args.training_input,
        args.validation_fraction,
        args.output,
        args.config,
        args.structure,
        args.learning_rate,
        args.regularizer,
        #args.momentum,
        #args.batch,
        ##args.min_epochs,
        #args.max_epochs,
        ##args.patience_increase,
        ##args.threshold,
        #args.min_delta,
        #args.patience,
        #args.profile,
        #args.nbworkers,
        #args.save_all,
        args.verbose
    )


if __name__ == '__main__':
    _main()

