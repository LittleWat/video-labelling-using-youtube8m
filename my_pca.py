# coding: UTF-8
from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse, json, os, glob

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import Variable
import numpy as np
from chainer.datasets import tuple_dataset


class PCAE(chainer.Chain):
    
    def __init__(self, n_inputs=2048, n_units=1024, c=1.0):
        super(PCAE, self).__init__(
            enc_l = L.Linear(n_inputs, n_units),  
            dec_l = L.Linear(n_units, n_inputs),
        )
        
        self.n_units = n_units
        self.c = c
        
        
    def __call__(self, x, pcaed_x=None, test=False):
        h = self.enc_l(x)
        rec = self.dec_l(h)
        
        if test:
            return rec, h        
        
        ae_loss = F.mean_squared_error(x, rec) 
        pca_loss = F.mean_squared_error(pcaed_x, h) * self.c
        
#         self.loss = ae_loss + pca_loss
        self.loss = pca_loss
        
        chainer.reporter.report({'loss': self.loss,
                                'ae_loss': ae_loss,
                                'pca_loss': pca_loss,
                                }, self)        

        
        return self.loss

def get_pcaed_feature(feature_2048_mat, model_path='pca_model/pcae.npz'):
    x_size = 2048
    h_size = 1024

    model = PCAE(x_size, h_size, 0.00001)

    chainer.serializers.load_npz(model_path, model)
    rec, h = model(feature_2048_mat, test=True)

    pca_feature = h.data.astype(np.int32)
    
    return pca_feature
