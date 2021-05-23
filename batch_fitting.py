import os, sys
import util
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
from photometric_fitting import PhotometricFitting


if __name__ == '__main__':
    import importlib
    import torch
    imagepath = str(sys.argv[1])
    maskpath = str(sys.argv[2])
    landmarkpath = str(sys.argv[3])
    device_name = str(sys.argv[4])
    if len(sys.argv)>5:
        model_name = str(sys.argv[5])
    else:
        model_name = 'flame'

    model_filename = "conf." + model_name
    modellib = importlib.import_module(model_filename)
    config = modellib.config

    config = util.dict2obj(config)
    util.check_mkdir(config.savefolder)

    config.batch_size = 1
    fitting = PhotometricFitting(config, device=device_name)

    fitting.batch_run(imagepath, maskpath, landmarkpath)
