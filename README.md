# A deep model for human activity recognition and mobile optimisation

CS7CS5 - M.Sc. Dissertation Project

## Table of content

- [Introduction](#Introduction)
- [Setup Environment](#Setup-Environment)
- [Running](#Running)

## Introduction

## Setup Environment

CPU only anaconda environment

```bash
conda env create --name DProject -f env-cpu.yaml
```

If you need to use GPU

```bash
conda env create --name DProject -f env-gpu.yaml
```

Download face detect and landmark models [HaarCascade](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml) and [FacemarkLBF](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml) to be used in the data set anonymisation algorithm.

## Running
