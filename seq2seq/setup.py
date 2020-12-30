#!/usr/bin/env python
#
# Copyright (c) 2020 The Ruse Authors.
#
"""Install Ruse Seq2seq model."""

import setuptools


def setup_package():
  long_description = "seq2seq"

  setuptools.setup(
      name='seq2seq',
      version='0.0.1',
      description='Universal Sentence Encoder',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Google Inc.',
      author_email='no-reply@google.com',
      url='http://github.com/google-research/ruse',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      install_requires=[
        'sentencepiece == 0.1.91',
        'transformers==3.5.1',
        'tensorboard',
        'scikit-learn',
        'seqeval',
        'psutil',
        'sacrebleu',
        'rouge-score',
        'tensorflow_datasets',
        'pytorch-lightning==1.0.4',
        'matplotlib',
        'git-python==1.0.3',
        'faiss-cpu',
        'streamlit',
        'elasticsearch',
        'nltk',
        'pandas',
        'datasets',
        'fire',
        'pytest',
        'conllu',
        'tf-nightly',
        'google-cloud-storage',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='text nlp machinelearning',
  )


if __name__ == '__main__':
  setup_package()
