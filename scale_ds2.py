#-*- coding: utf-8 -*-
# Copyright 2022 István Z. Kovács. All Rights Reserved.
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
# ==============================================================================
"""Script to train with TensorFlow Cloud"""

import os
import tensorflow as tf
import tensorflow_cloud as tfc

import trainer.ds2_train as ds2_train

# Run on TF Cloud
tfc.run(
    entry_point=None,
    distribution_strategy=None,
    worker_count=0,
    docker_config=tfc.DockerConfig(image_build_bucket=ds2_train.GCP_BUCKET),
    requirements_txt="requirements.txt",
    stream_logs=True,
)

# Run training
ds2_train.main_tfc()

