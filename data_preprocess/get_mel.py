# -*- coding: utf-8 -*-

import numpy as np

from audio_driven.models.audio_encoder import *

audio_file = 'test.wav'
save_path = ''
source_audio_feature, source_nums = get_mel(audio_file)
np.save(save_path, source_audio_feature)