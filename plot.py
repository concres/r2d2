import numpy as np 
import cv2, os
import math
import matplotlib.pyplot as plt
# import time

sequence_num = "04" # two digits

root_path = "/media/alves/alves32/dataset/"
# root_path  = "/home/thiago/verlab/projects/xquad/code/datasets/testes_slam/kitti/"

poses_path = root_path+"poses/"+sequence_num+".txt"	
with open(poses_path) as f:
	gt_info = f.readlines()

gt_pose = np.empty((0,3), dtype=np.float32)
for line in gt_info:
	ss = line.strip().split()
	gt_pose = np.append(gt_pose, [[float(ss[3]), float(ss[7]), float(ss[11])]], axis=0)

est_pose = np.load('evaluation/feat/traj_seq_'+sequence_num+'.npy')
rf_pose = np.load('evaluation/rf/traj_seq_'+sequence_num+'.npy')
net_pose = np.load('evaluation/net/traj_seq_'+sequence_num+'.npy')
gt_pose = gt_pose[-len(est_pose):] # remove the firsts poses

max_pose = np.empty((0,3), dtype=np.float32)
min_pose = np.empty((0,3), dtype=np.float32)

# print(max_pose.shape)
# print([np.amax(gt_pose, axis=0)].shape)

max_pose = np.append(max_pose, [np.amax(gt_pose, axis=0)], axis=0)
min_pose = np.append(min_pose, [np.amin(gt_pose, axis=0)], axis=0)

max_pose = np.append(max_pose, [np.amax(est_pose, axis=0)], axis=0)
min_pose = np.append(min_pose, [np.amin(est_pose, axis=0)], axis=0)

max_pose = np.append(max_pose, [np.amax(rf_pose, axis=0)], axis=0)
min_pose = np.append(min_pose, [np.amin(rf_pose, axis=0)], axis=0)

max_pose = np.append(max_pose, [np.amax(net_pose, axis=0)], axis=0)
min_pose = np.append(min_pose, [np.amin(net_pose, axis=0)], axis=0)

maxx = np.amax(max_pose, axis=0)
minn = np.amin(min_pose, axis=0)

# window size to show the map
border = 10
scale = 1.0
# window_width  = int(abs(max_x - min_x)*scale + border)
# window_height = int(abs(max_z - min_z)*scale + border)
window_width  = int(abs(maxx[0] - minn[0])*scale + border)
window_height = int(abs(maxx[2] - minn[2])*scale + border)

# starting points for drawing functions
w_start = int(-minn[0]*scale + (border/2))
h_start = int(-minn[2]*scale + (border/2))


# erros =  np.load('error_seq_'+sequence_num+'.npy')
# pred_traj = np.load('traj_seq_'+sequence_num+'.npy')

traj = np.zeros((window_height,window_width,3), dtype=np.uint8)
for i in range(len(gt_pose)):
	gt_x, gt_y, gt_z = gt_pose[i]
	est_x, est_y, est_z = est_pose[i]
	rf_x, rf_y, rf_z = rf_pose[i]
	net_x, net_y, net_z = net_pose[i]

	
	cv2.circle(traj, (int((gt_x +w_start)*scale), int((gt_z +h_start)*scale)), 1, (0,255,0), -1) # ground truth
	cv2.circle(traj, (int((est_x+w_start)*scale), int((est_z+h_start)*scale)), 1, (255,0,0), -1) # estimate
	cv2.circle(traj, (int((rf_x+w_start)*scale), int((rf_z+h_start)*scale)), 1, (0,0,255), -1) # random forest
	cv2.circle(traj, (int((net_x+w_start)*scale), int((net_z+h_start)*scale)), 1, (0,255,255), -1) # neural network

cv2.imwrite('plots/trajs_' + sequence_num + '.png', traj)
# cv2.imshow('Trajectory Sequence '+sequence_num, traj)
# cv2.waitKey(1)


error_feat =  np.load('evaluation/feat/error_seq_'+sequence_num+'.npy')
error_rf =  np.load('evaluation/rf/error_seq_'+sequence_num+'.npy')
error_net =  np.load('evaluation/net/error_seq_'+sequence_num+'.npy')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(error_feat)), error_feat, s=1, c='b', label='Feature')
ax1.scatter(range(len(error_rf)), error_rf, s=1, c='r', label='Feature + RFC')
ax1.scatter(range(len(error_net)), error_net, s=1, c='g', label='Feature + neural')
plt.legend(loc='upper left');
plt.ylabel('Error meters')
plt.savefig('plots/error_' + sequence_num + '.png')
