"""Projections of measured features; shared intersections are negative controls."""
import numpy as np
from ct_lab.filtering import projection,fit_scaler,standardize


def project(values,fold,kind='PCA'):
    if values.shape[1]==1 and kind=='PCA':
        scaler=fit_scaler(values[fold]);x=standardize(values,scaler)[:,0]
        return np.column_stack([x,np.zeros(len(x))]),{'scaler':scaler,'settings':{'method':'PCA','components':[[1.],[0.]],
                    'explained_variance_ratio':[1. if np.std(x[fold]) else 0.,0.],'one_dimensional_input':'zero second coordinate'},'fit_scope':'train only'}
    return projection(values,fold,kind)
