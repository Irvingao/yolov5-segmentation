# General utils

import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

