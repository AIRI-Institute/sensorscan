# SensorSCAN: Self-Supervised Learning and Deep Clustering for Fault Diagnosis in Chemical Processes


## Installation

```
$ conda create --solver=libmamba -n SensorSCAN -c rapidsai -c conda-forge -c nvidia rapids=23.08 python=3.9 cuda-version=11.8
$ conda activate SensorSCAN
$ pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install git+https://github.com/airi-industrial-ai/fddbenchmark
$ pip install hydra-core==1.3
$ pip install numpy==1.22.2 --no-deps
```

## Self-Supervised Pretraining
For TEP Rieth dataset run the following command:
```
$ conda activate SensorSCAN
$ python ssl_pretraining.py --config-name rieth
```

For TEP Ricker dataset run the following command:
```
$ conda activate SensorSCAN
$ python ssl_pretraining.py --config-name reinartz
```


## SCAN Training
For TEP Rieth dataset run the following command:
```
$ conda activate SensorSCAN
$ python SCAN_training.py --config-name  scan_rieth.yaml +pretrained_path=<path to the model trained in the previous step>
```

For TEP Ricker dataset run the following command:
```
$ conda activate SensorSCAN
$ python SCAN_training.py --config-name  scan_reinartz.yaml +pretrained_path=<path to the model trained in the previous step>
```

## Evaluation
For TEP Rieth dataset run the following command:
```
$ conda activate SensorSCAN
$ python evaluate.py --config-name  scan_rieth.yaml +encoder_path=<path to the encoder weights> +clustering_path=<path to clustering model weights>
```

For TEP Ricker dataset run the following command:
```
$ conda activate SensorSCAN
$ python evaluate.py --config-name  scan_reinartz.yaml +encoder_path=<path to the encoder weights> +clustering_path=<path to clustering model weights>
```
