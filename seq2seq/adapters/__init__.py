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
from .adapter_configuration import ADAPTER_CONFFIG_MAPPING, AutoAdapterConfig
from .adapter_configuration import AdapterConfig, MetaAdapterConfig, ParametricMetaAdapterConfig
from .adapter_controller import AdapterController, MetaAdapterController, AutoAdapterController
from .adapter_modeling import Adapter
from .adapter_utils import HyperNetDownSampler, HyperNetUpSampler, TaskEmbeddingController