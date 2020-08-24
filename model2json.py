import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import theano
from theano import tensor as T, function, printing
import keras.backend as K
import h5py
import keras.models
import argparse

from datetime import datetime
today = datetime.date(datetime.now())


np.random.seed(12122)
keras.activations.abs = K.abs


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",    type=str,
                        help="Input model file (h5) with full path")
    parser.add_argument("--outdir",       type=str,  default="./MDN_models_%s"%today,
                        help="Path to the output directory")
    parser.add_argument("--nparticle",       type=int,
                        help="Number of particle: 1, 2, 3")
    args = parser.parse_args()

    inputFile        = args.infile
    outputDir        = args.outdir
    nHits            = args.nparticle

    print("=====================================================================")
    print("INPUTS to the SCRIPT: "                                               )
    print("=====================================================================")
    print("inputFile             : ", inputFile                                  )
    print("outputDir             : ", outputDir                                  )
    print("nParticles            : ", nHits                                      )
    print("=====================================================================")

    comd = "mkdir -p "+outputDir
    os.system(comd)

    if nHits == 1:
        pos_suff = 'pos1'
    elif nHits == 2:
        pos_suff = 'pos2'
    elif nHits == 3:
        pos_suff = 'pos3'

    model = keras.models.load_model(
            inputFile, 
            custom_objects={
                'loss': mixture_density_loss(nb_components=1)
            }
        )

    # get the architecture as a json string
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputDir+'/architecture_'+pos_suff+'.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputDir+'/weights_'+pos_suff+'.h5')



# when calling this script
if __name__ == "__main__":
    main()

