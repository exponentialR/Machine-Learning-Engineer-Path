# import os
# import numpy as np
# import sys 

# time_per_action = 1
# DATA_PATH = os.path.join('folder_flow') 

# actions = np.array(['intent', 'thanks', 'iloveyou'])
# import time
# from time import perf_counter

# # Thirty videos worth of data
# no_sequences = 30

# # Videos are going to be 30 frames in length
# sequence_length = 30
# start_time = None
# # # Folder start
# start_folder = 30
# # for action in actions: 
# #     for sequence in range(no_sequences):
# #         try: 
# #             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
# #         except:
# #             pass
# action_count = 0
# time_check = []
# frame_num = 1
# # start_= perf_counter()
# sequence_count = 0
# running = True


# if not running:
#     sys.exit
# start_= perf_counter()#time.time()
# if start_time is None: 
#     start_time = start_
# passed_time = start_ - start_time

# for i in range(sequence_length):
#     if passed_time <= time_per_action:
#         action = actions[action_count]
#         time_check.append(passed_time)
#         keypoints = np.random.rand(126,2).flatten()
#         file_name = f'keypoints_{frame_num}.npy'
        
#         npy_path = os.path.join(DATA_PATH, actions[action_count],str(i), str(file_name))
#         np.save(npy_path, keypoints)
#         frame_num +=1
#     else:    
#         time_file_name = 'time_since_last_action_start.npy'
#         path_to_time_file = os.path.join(DATA_PATH, actions[action_count], str(i), time_file_name)
#         np.save(path_to_time_file, time_check)
#         time_check = []
#         frame_num = 1
#         start_time = None
#         try:
        
        
#             if action_count<len(actions) and i>=30:
#                 action_count +=1
#                 running = True
#             elif action_count > len(actions) and i>=30:
#                 running = False
#             else:
#                 pass
#         except:KeyError

import numpy as np 
np_load_old = np.load
np.load = lambda*a, **k: np_load_old(*a, allow_pickle=True, **k)
y = np.load(r"E:\20.01.22\H-E-O\handover_intention\1_handover1\4\9.npy")
m = []
if y.any() == None:
    y = np.zeros(132).tolist()
    m.append(y)
    m.append(np.zeros(10))
    print(m)
    print(m.shape)
    # y = y.flatten()
    # print(y.shape)