

* import sys
import logging
import itertools

import numpy as np
np.random.seed(0)
import gym

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')