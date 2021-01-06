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

def string_to_float(string, default=-1.):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def string_to_int(string, default=-1):
    """Converts string to int, using default when conversion not possible."""
    try:
        return int(string)
    except ValueError:
        return default


def get_post_processor(task):
    """Returns post processor required to apply on the predictions/targets
    before computing metrics for each task."""
    if task == "stsb":
        return string_to_float
    elif task in ["qqp", "cola", "mrpc"]:
        return string_to_int
    else:
        return None
