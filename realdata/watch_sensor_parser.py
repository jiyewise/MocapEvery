import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import dateutil.parser
from IPython import embed
from scipy import ndimage 
from copy import deepcopy
from fairmotion.ops import conversions
import matplotlib.pyplot as plt
import numpy as np
from realdata.utils import * 
from scipy.signal import find_peaks

g2ms2 = 9.81
pd.set_option('display.float_format', lambda x: '%.6f' % x)
imu_smooth_window = 3
average_windows = imu_smooth_window * 2 + 1
watch_record_time_offset_default = 0.18 # NOTE why does this offset change?!?!
watch_acc_sign = -1

def interpolate(t1, v1, t2, v2, t):
    time_diff = (t2 - t1) + 1e-06
    fraction = (t - t1) / time_diff
    v = v1 + fraction * (v2 - v1)
    return v

def resample(real_timecode, imu_timecode, imu_ori, imu_acc):
    resampled_imu_ori = []
    resampled_imu_acc = []
    for i, t_real in enumerate(real_timecode):
        imu_idx = (np.abs(imu_timecode - t_real)).argmin()
        neighbor_imu_flag = 1 if imu_timecode[imu_idx] < t_real else -1
        
        # interpolate
        if neighbor_imu_flag > 0:
            if imu_idx + neighbor_imu_flag >= len(imu_acc):
                acc = imu_acc[imu_idx]
            else:
                acc = interpolate(imu_timecode[imu_idx], imu_acc[imu_idx],  imu_timecode[imu_idx+neighbor_imu_flag], imu_acc[imu_idx+neighbor_imu_flag], t_real)
        else:
            acc = interpolate(imu_timecode[imu_idx+neighbor_imu_flag], imu_acc[imu_idx+neighbor_imu_flag], imu_timecode[imu_idx], imu_acc[imu_idx],  t_real)

        resampled_imu_ori.append(imu_ori[imu_idx]) # TODO slerp
        resampled_imu_acc.append(acc)

    return np.array(resampled_imu_ori), np.array(resampled_imu_acc)

class WatchIMUSignal2(object):
    def __init__(self, csv_file, lr, time_offset=None) -> None:
        if ".csv" in csv_file:
            if time_offset is not None:
                self.time_offset = time_offset / 30.0
            else:
                self.time_offset = 0.0
            self.load_from_csv(csv_file=csv_file, lr=lr)
        else:
            print("wrong csv file")
            embed()

    def load_from_csv(self, csv_file, lr):
        self.lr = lr
        df = pd.read_csv(csv_file)
        # Convert loggingTime to datetime and then to Unix timestamp
        timeseries = pd.to_datetime(df['loggingTime(txt)']).apply(lambda x: x.timestamp()) 
        self.timecode = timeseries.to_numpy() + self.time_offset
        # self.timecode = timeseries.to_numpy() + watch_record_time_offset_default + self.time_offset

        self.original_acc = df[['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']].to_numpy() * g2ms2 * watch_acc_sign
        self.user_acc = df[['motionUserAccelerationX(G)', 'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)']].to_numpy() * g2ms2 *watch_acc_sign

        self.euler_ori = df[['motionPitch(rad)', 'motionRoll(rad)', 'motionYaw(rad)']].to_numpy()
        self.quat_ori = df[['motionQuaternionX(R)', 'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)']].to_numpy()

        self.filter_signals()

    def update_timecode(self, time_offset):
        # return
        print(f"time offset: {time_offset}")
        self.timecode = self.timecode + time_offset/30.0

    def filter_signals(self):
        # self.quat_ori = ndimage.uniform_filter1d(deepcopy(self.quat_ori), size=imu_smooth_window, axis=0)

        self.original_acc = ndimage.uniform_filter1d(deepcopy(self.original_acc), size=imu_smooth_window, axis=0)
        self.user_acc = ndimage.uniform_filter1d(deepcopy(self.user_acc), size=imu_smooth_window, axis=0,  mode="nearest")
        # self.user_acc, _ = self.kalman_filter(self.user_acc)
        # self.user_acc = ndimage.gaussian_filter1d(deepcopy(self.user_acc), sigma=1, axis=0)
        

    def calib_origin(self, origin_timecode_range):
        lower = origin_timecode_range[0]
        upper = origin_timecode_range[1]
        # mask = (self.timecode >= lower) & (self.timecode <= upper)
        origin_indices =  np.where((self.timecode >= lower) & (self.timecode <= upper))
        origin_ori = np.mean(self.quat_ori[origin_indices], axis=0)
        origin_ori_R = conversions.Q2R(origin_ori)

        # embed()
        self.R_ori = conversions.Q2R(self.quat_ori) 
        R = origin_ori_R.transpose()
        # self.R_ori_calibrated = origin_ori_R.transpose() @ self.R_ori
        self.R_ori_calibrated = R @ self.R_ori

        # embed()
        # self.original_acc_calibrated = np.einsum('ijk,ik->ij', self.R_ori_calibrated, self.original_acc)
        self.user_acc_calibrated = np.einsum('ijk,ik->ij', self.R_ori_calibrated, self.user_acc)

        # self.user_acc_global = np.einsum('ijk,ik->ij', self.R_ori, self.user_acc)
        # self.original_acc_global = np.einsum('ijk,ik->ij', self.R_ori, self.original_acc)

        # self.original_acc_calibrated = np.einsum('ij,kj->ki', R, self.original_acc_global)
        # self.user_acc_calibrated = np.einsum('ij,kj->ki', R, self.user_acc_global)


    def calib_tpose(self, tpose_timecode_range, tpose_joint_ori):
        lower = tpose_timecode_range[0]
        upper = tpose_timecode_range[1]

        tpose_indices =  np.where((self.timecode >= lower) & (self.timecode <= upper))

        tpose_R_list = self.R_ori_calibrated[tpose_indices]
        tpose_A = conversions.R2A(tpose_R_list)
        tpose_R_mean = conversions.A2R(np.mean(tpose_A, axis=0)) # TODO check

        imu2joint = tpose_R_mean.transpose() @ tpose_joint_ori

        self.imu2joint = imu2joint


    def parse_and_resample2(self, start_timecode, end_timecode, frame_num, gt_timecode_list):
        frames = time_range_into_frames(self.timecode, start_timecode=start_timecode, end_timecode=end_timecode)

        imu_timecode, imu_ori, imu_acc = self.timecode[frames], self.R_ori_calibrated[frames], self.user_acc_calibrated[frames]
        
        # embed()
        resampled_imu_ori, resampled_imu_acc = resample(gt_timecode_list, imu_timecode, imu_ori, imu_acc)
        self.imu_ori = resampled_imu_ori[:frame_num]
        self.imu_acc = resampled_imu_acc[:frame_num]
    

    def imu_ori_to_joint(self):
        self.imu_ori = self.imu_ori @ self.imu2joint


    def parse_and_resample(self, start_timecode, frame_num, gt_timecode_list):
        imu_timecode, imu_ori, imu_acc = self.get_sensor_data_within_range_frames(start_timecode, frame_num)
        # embed()
        resampled_imu_ori, resampled_imu_acc = resample(gt_timecode_list, imu_timecode, imu_ori, imu_acc)
        self.imu_ori = resampled_imu_ori[:frame_num]
        self.imu_acc = resampled_imu_acc[:frame_num]


    def get_sensor_data_within_range_frames(self, start_timecode, frame_num):
        timecode_indices = np.where(self.timecode >= start_timecode)[0]
        timecode_indices = timecode_indices[:frame_num+100]
        return self.timecode[timecode_indices], self.R_ori_calibrated[timecode_indices], self.user_acc_calibrated[timecode_indices]


    def get_sensor_data_within_range_timecode(self, start_timecode, end_timecode):
        indices = time_range_into_frames(self.timecode, start_timecode=start_timecode, end_timecode=end_timecode)
        return self.timecode[indices], self.R_ori_calibrated[indices], self.user_acc_calibrated[indices]
    

    def get_mid_peak_from_clap_range(self, clap_frames):
        acc_within_range = self.user_acc_calibrated[clap_frames[0]:clap_frames[-1]] # [#F in clap range, 3]
        peaks, properties = find_peaks(acc_within_range[:,1], distance=1, prominence=5, height=0)
        # embed()
        if len(peaks) % 3 != 0:
            print(f"Please check peaks! current function: get_mid_peak_from_clap_range in WatchIMUSignal")
            embed()
        if len(peaks) == 1:
            return peaks[0]
        return peaks[1]

    # def time_range_into_frames(self, start_timecode, end_timecode):
    #     return np.where((self.timecode >= start_timecode) & (self.timecode <= end_timecode))[0]
        
if __name__ == "__main__":
    csv_filepath = "../data/realdata/0707/watch_sensorlog.csv"
    watch_signal = WatchIMUSignal2(csv_file=csv_filepath)
    watch_signal.calib_origin([1688721370.032988, 1688721373.0479887])

    

# # Read the csv file
# df = pd.read_csv('file.csv')

# # Convert loggingTime to datetime and then to Unix timestamp
# df['loggingTime'] = df['loggingTime'].apply(lambda x: dateutil.parser.parse(x).timestamp())

# # Define your time range
# time_start = datetime(2023, 7, 7, 12, 0, 0, tzinfo=timezone.utc).timestamp() # adjust date-time as per your needs
# time_end = datetime(2023, 7, 7, 19, 0, 0, tzinfo=timezone.utc).timestamp() # adjust date-time as per your needs

# # Select rows within specific range
# selected_rows = df[(df['loggingTime'] >= time_start) & (df['loggingTime'] <= time_end)]

# # Get accelerometerAccelerationX, Y, Z values of the selected rows
# selected_accelerations = selected_rows[['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']]

# print(selected_accelerations)