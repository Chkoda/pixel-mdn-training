import _thread
import threading

import numpy as np
import ROOT
import root_numpy as rnp

import time

##########################################################################################################################
def stopWatch(value):
    '''From seconds to Days;Hours:Minutes;Seconds'''

    valueD = (((value/365)/24)/60)
    Days = int (valueD)

    valueH = (valueD-Days)*365
    Hours = int(valueH)

    valueM = (valueH - Hours)*24
    Minutes = int(valueM)

    valueS = (valueM - Minutes)*60
    Seconds = int(valueS)


    print(Days,";",Hours,":",Minutes,";",Seconds)



##########################################################################################################################


def calc_normalization(path, tree, branches):
    tfile = ROOT.TFile(path, 'READ')
    tree = tfile.Get(tree)
    mean = np.zeros(len(branches))
    std = np.ones(len(branches))
    for i, branch in enumerate(branches):
        arr = rnp.tree2array(tree, branches=branch)
        mean[i] = np.mean(arr)
        std[i] = np.std(arr)
        if std[i] == 0:
            std[i] = 1
    return {'std': std, 'mean': mean}


# http://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
def root_batch(tree, branches, bounds, normalization):
    batch = rnp.tree2array(
        tree=tree,
        branches=branches,
        start=bounds[0],
        stop=bounds[1]
    )
    
    #print("MY batch is: ")
    #print(batch)
    #print("Btch Shape:  ")
    #print(batch.shape)
    batch = batch.view(np.float64).reshape(batch.shape + (-1,))
    batch -= normalization['mean']
    batch *= (1.0/normalization['std'])
    #print("Batches again: ")
    #print(batch)
    return batch




# https://github.com/fchollet/keras/issues/1638#issuecomment-179744902
'''class ThreadsafeIter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, itr):
        self.itr = itr
        self.lock = threading.Lock()
        #print(self.lock)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock: 
            #print(self.lock)
            return self.itr.__next__()


def threadsafe_generator(gen):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    print("Here First =============================== Elham")
    def safegen(*a, **kw):
        print("Second======================== Elham")
        return ThreadsafeIter(gen(*a, **kw))
    return safegen

'''

#def generator(path,
#              tree,
#              branches,
#              batch=32,
#              normalization=None,
#              train_split=1,
#              loop=True): \
#              # pylint: disable=too-many-arguments
#    print("Hello ................................")
#	
#    tfiles = {}
#    trees = {}
#
#    ntrain = int(train_split * get_entries(path, tree))
#
#    if normalization is None:
#        normalization = {'std': 1, 'mean': 0}
#
#    mytfile = ROOT.TFile(path, 'READ')               
#    mytree = mytfile.Get(tree)
#    
#    print('preparing input numpy array')
#    xset = root_batch(
#        tree=mytree,
#        branches=branches[0],
#        bounds=(0, ntrain),
#        normalization=normalization
#        )
#    print('preparing target numpy array')
#    yset = root_batch(
#        tree=mytree,
#        branches=branches[1],
#        bounds=(0, ntrain),
#        normalization={'mean': 0, 'std': 1}
#        )
#
#    print('done preparing nympy arrays') 
#
#    while True:
#        for i in range(0, ntrain, batch):
#            #print("Now at batch starting at %i" % (i))
#            if len(branches[1]) > 0:
#                #print("yielding ")
#                yield (xset[i:(i+batch),:], yset[i:(i+batch)])
#            else:
#                yield (xset[i:(i+batch),:],None)
#
#                        
#        if not loop:
#            break
#@threadsafe_generator
def generator(path,
              tree,
              branches,
              batch=32,
              normalization=None,
              train_split=1,
              loop=True): \
              # pylint: disable=too-many-arguments
    tfiles = {}
    trees = {}
    ntrain = int(train_split * get_entries(path, tree))
    if normalization is None:
        normalization = {'std': 1, 'mean': 0}
    while True:
        for i in range(0, ntrain, batch):
            thr = _thread.get_ident()
            if thr not in trees:
                tfiles[thr] = ROOT.TFile(path, 'READ')
                trees[thr] = tfiles[thr].Get(tree)
            xbatch = root_batch(
                tree=trees[thr],
                branches=branches[0],
                bounds=(i, i+batch),
                normalization=normalization
            )
            if len(branches[1]) > 0:
                ybatch = root_batch(
                    tree=trees[thr],
                    branches=branches[1],
                    bounds=(i, i+batch),
                    normalization={'mean': 0, 'std': 1}
                )
            else:
                ybatch = None
            yield (xbatch, ybatch)
        if not loop:
            break





def load_train_set(path,tree, branches,normalization=None, train_split=1,):

    tfile = ROOT.TFile(path, 'READ')
    mytree = tfile.Get(tree)

    if normalization is None:
        normalization = {'std': 1, 'mean': 0}

    #nentries = mytree.GetEntries()
    nentries = 100000 


    start = 0
    ntrain = int(nentries*train_split)
    #ntrain = int(train_split * get_entries(path, mytree))
    print("N_Train:", ntrain)
    print("Entries:  ", nentries)
    print("train_split: ", train_split)
    #ntrain = 100000
    xdat = root_batch(
        tree=mytree,
        branches=branches[0],
        bounds=(start, ntrain),
        normalization=normalization,
    )
    ydat = root_batch(
        tree=mytree,
        branches=branches[1],
        bounds=(start, ntrain),
        normalization={'mean': 0, 'std': 1}
    )

    print(ydat[:,0])
    return xdat, ydat



def load_validation(path, tree, branches, normalization, validation_split=0):

    newtfile = ROOT.TFile(path, 'READ')
    newtree = newtfile.Get(tree)

    if normalization is None:
        normalization = {'std': 1, 'mean': 0}

    nentries = newtree.GetEntries()
    start = int(nentries * (1 - validation_split))

    x_t = root_batch(
        tree=newtree,
        branches=branches[0],
        bounds=(start, nentries),
        normalization=normalization,
    )
    y_t = root_batch(
        tree=newtree,
        branches=branches[1],
        bounds=(start, nentries),
        normalization={'mean': 0, 'std': 1}
    )

    return x_t, y_t #[y_t[:,0],y_t[:,1]]

start = time.time()   #Elham
def get_entries(path, tree):
    tfile = ROOT.TFile(path, 'READ')
    ttree = tfile.Get(tree)
    return ttree.GetEntries()
end = time.time() 
#stopWatch(end-start)
