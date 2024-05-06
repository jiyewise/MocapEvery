"""
IMU sensor related
"""

# synthesis
imu_num = 2
imu_joint_names = ["LeftHand", "RightHand"]
joint_mask = []
vertex_mask = {}
vertex_mask[imu_joint_names[0]] = [4576, 4577, 4583, 4858] # LeftHand
vertex_mask[imu_joint_names[1]] = [7312, 7313, 7319, 7594] # RightHand

fps = 30.0
syn_frame_n = 4
syn_acc_dt = syn_frame_n * (1/fps)

# real world (apple watch sensors)
# up_axis = "y"