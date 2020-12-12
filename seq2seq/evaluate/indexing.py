"""Builds tools for indexing and search with FAISS."""

import faiss
import numpy as np


def IndexCreate(x, idx_type="FlatL2", normalize=True):
    assert idx_type == 'FlatL2', 'only FlatL2 index is currently supported'
    print(x.shape)
    dim = x.shape[1]
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
    idx.add(x)
    return idx


def IndexSearchMultiple(data, idx):
    """Search closest vector for all languages pairs and calculate error rate."""
    nl = len(data)
    nbex = data[0].shape[0]
    err = np.zeros((nl, nl)).astype(float)
    ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
    for i1 in range(nl):
        for i2 in range(nl):
            if i1 != i2:
                D, I = idx[i2].search(data[i1], 1)
                err[i1, i2] \
                        = (nbex - np.equal(I.reshape(nbex), ref)
                           .astype(int).sum()) / nbex
    return err


def IndexPrintConfusionMatrix(err, langs):
    """Print confusion matrix."""
    nl = len(langs)
    assert nl == err.shape[0], 'size of errror matrix doesn not match'
    print('Confusion matrix:')
    print('{:8s}'.format('langs'), end='')
    for i2 in range(nl):
        print('{:8s} '.format(langs[i2]), end='')
    print('{:8s}'.format('avg'))
    for i1 in range(nl):
        print('{:3s}'.format(langs[i1]), end='')
        for i2 in range(nl):
            print('{:8.2f}%'.format(100 * err[i1, i2]), end='')
        print('{:8.2f}%'.format(100 * err[i1, :].sum() / (nl-1)))
    print('avg', end='')
    for i2 in range(nl):
        print('{:8.2f}%'.format(100 * err[:, i2].sum() / (nl-1)), end='')
    # global average
    print('{:8.2f}%'.format(100 * err.sum() / (nl-1) / nl))