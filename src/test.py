import pickle

import numpy as np

from src.constants import MODEL_PATH
from src.utils import test_pre_processing

test_data = test_pre_processing('C:/Users/rosha/OneDrive/Desktop/MetroDx/test files/TEST FILES/sub10 kDa')
filename = MODEL_PATH + '/metrodx_model.sav'
model_pipe = pickle.load(open(filename, 'rb'))
prediction = model_pipe.predict(test_data)