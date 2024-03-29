# DeepSpeech2 Model fork [05.04.2022]
![Exp](https://img.shields.io/badge/Fork-experimental-orange.svg)
[![Lic](https://img.shields.io/badge/License-Apache2.0-green)](http://www.apache.org/licenses/LICENSE-2.0)
![Py](https://img.shields.io/badge/python-3.7+-green)
[![TensorFlow 2.9](https://img.shields.io/badge/TensorFlow-2.9-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.9.0)
[![Keras](https://img.shields.io/badge/KerasAPI-OK-green)](https://keras.io/api/)
![GC](https://img.shields.io/badge/GoogleCloud_VertexAI-OK-green)
![Docker](https://img.shields.io/badge/Docker-OK-green)
![Ver](https://img.shields.io/badge/version-0.3-lightgrey)

This is a stand-alone fork for the Tensorflow [DeepSpeech2](https://github.com/tensorflow/models/tree/master/research/deep_speech) code.

Development plans & status:

- [x] Updates to work with [tf-models-official](https://pypi.org/project/tf-models-official/) python package and with TF2.8
- [x] Updates to use [absl-py](https://abseil.io/docs/python/) python package
- [x] Updates to work with [Common Voice Corpus 2](https://commonvoice.mozilla.org/en/datasets) 
- [x] Refactor code to use Keras Functional API
- [x[ Update ReadMe for the use of the new code which uses Keras Functional AP
- [x] Modularize to match custom training requirements on GC with Vertex AI
- [x] Local training test (Python 3.8+)
- [x] Custom training job in custom container on GC (Vertex AI, Artifacts Registry, etc.)
- [x] Custom training in VertexAI/Workbench, JupyterLab notebook (VM instance with 1x GPU T4)
- [x] Fine-tuned model training performance
- [ ] Remove old code which uses tf.estimator, etc.
- [ ] Compile and make it work with [Rahsspy](https://github.com/rhasspy/rhasspy), as replacement for the [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)




# Original ReadMe: DeepSpeech2 Model

![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
[![TensorFlow 1.15.3](https://img.shields.io/badge/TensorFlow-1.15.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.3)
[![TensorFlow 2.3](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)

This is an implementation of the [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) model. Current implementation is based on the code from the authors' [DeepSpeech code](https://github.com/PaddlePaddle/DeepSpeech) and the implementation in the [MLPerf Repo](https://github.com/mlperf/reference/tree/master/speech_recognition).

