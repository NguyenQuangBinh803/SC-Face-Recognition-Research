import numpy as np

frame = np.zeros((500,500))
thermal_data = 36.5


global_locs = []
global_mask = []

face_detect_status = False
mask_detect_status = False
human_appear_status = False


face_area = 0