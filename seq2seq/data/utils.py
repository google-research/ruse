# Copyright 2020 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

def round_stsb_target(label):
  """STSB maps two sentences to a floating point number between 1 and 5
  representing their semantic similarity. Since we are treating all tasks as
  text-to-text tasks we need to convert this floating point number to a string.
  The vast majority of the similarity score labels in STSB are in the set
  [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
  entry in this set, and then we convert the result to a string (literally e.g.
  "3.4"). This converts STSB roughly into a 26-class classification dataset.
  Args:
    label: original label.
  Returns:
    A preprocessed label.
  """
  return np.round((label * 5) / 5, decimals=1)