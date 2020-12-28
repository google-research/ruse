# Lint as: python3
"""TODO(rabeeh): DO NOT SUBMIT without one-line documentation for transfer.

TODO(rabeeh): DO NOT SUBMIT without a detailed description of transfer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  app.run(main)
