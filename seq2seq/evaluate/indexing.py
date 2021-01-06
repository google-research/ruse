# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Builds tools for indexing and search with FAISS library."""

import faiss
import numpy as np


def IndexCreate(x, idx_type="FlatL2", normalize=True):
    """Creates an index over the given dataset."""
    assert idx_type == 'FlatL2', 'only FlatL2 index is currently supported'
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
    """Print confusion matrix across all the languages."""
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
    # Shows the global average.
    print('{:8.2f}%'.format(100 * err.sum() / (nl-1) / nl))