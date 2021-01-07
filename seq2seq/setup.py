#!/usr/bin/env python
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
