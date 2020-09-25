import numpy as np 
import cv2, os
import math
import matplotlib.pyplot as plt

from visual_odometry import PinholeCamera, VisualOdometry

sequence_num = "04" # two digits

# root_path = "/media/alves/alves32/dataset/"
root_path  = "/home/thiago/verlab/projects/xquad/code/datasets/testes_slam/kitti/"

img_dir    = root_path+"sequences/"+sequence_num+"/image_0/"
poses_path = root_path+"poses/"+sequence_num+".txt"	

# get camera parameters from projection matrix P0 (3x4)
f = open(root_path+"sequences/"+sequence_num+"/calib.txt")
P0 = f.readline()
P0 = P0.replace('\n', '').replace('P0: ','').split(' ')
P = np.array([float(x) for x in P0]).reshape((3,4))

fx 		= P[0,0] # focal length X
fy 		= P[1,1] # focal length Y
cx 		= P[0,2] # principal point X
cy 		= P[1,2] # principal point Y

# get img width, height
img_path = img_dir+"000000.png" 
img = cv2.imread(img_path, 0)

height 	= img.shape[0]
width 	= img.shape[1]

# create Camera and VisualOdometry objects

#cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
cam = PinholeCamera(width, height, fx, fy, cx, cy)
vo_learning = VisualOdometry(cam, poses_path, learning=True)
vo_fast 	= VisualOdometry(cam, poses_path, learning=False)

f2 = open(poses_path, 'r')
lines = f2.readlines()

max_x = max_y = max_z = -np.inf
min_x = min_y = min_z =  np.inf
tx = ty = tz = 0

for line in lines:
	T = np.array([float(x) for x in line.replace('\n', '').split(' ')]).reshape((3,4)) # 3x3 rotation matrix + 3x1 translation
	tx = T[0,3]
	ty = T[1,3]
	tz = T[2,3]
	if tx > max_x: max_x = tx
	if ty > max_y: max_y = ty
	if tz > max_z: max_z = tz
	
	if tx < min_x: min_x = tx
	if ty < min_y: min_y = ty
	if tz < min_z: min_z = tz
	
# window size to show the map
w_extra = 290
h_extra = 210
window_width  = int(abs(max_x - min_x)*1.5 + w_extra)
window_height = int(abs(max_z - min_z)*1.5 + h_extra)

# starting points for drawing functions
w_start = int(window_width/2)
h_start = int(window_height/10)+60

print("window_width, window_height = ", window_width, window_height)

traj = np.zeros((window_height,window_width,3), dtype=np.uint8)
errors = np.empty((1,0), dtype=np.float32)

num_imgs = len([img_name for img_name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img_name))])

for img_id in range(num_imgs):

	img_path = img_dir+str(img_id).zfill(6)+".png"
	img = cv2.imread(img_path, 0)

	vo_learning.update(img, img_id, img_path)
	vo_fast.update(img, img_id, img_path)

	cur_t = vo_learning.cur_t
	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.

	draw_x, draw_y = int(x)			, int(z)
	
	cur_t = vo_fast.cur_t
	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.

	draw_x_fast, draw_y_fast = int(x)			, int(z)
	
	true_x, true_y = int(vo_learning.trueX)	, int(vo_learning.trueZ)

	error = math.sqrt((vo_learning.trueX - x)**2 + (vo_learning.trueY - y)**2 + (vo_learning.trueZ - z)**2)
	errors = np.append(errors, error)

	cv2.circle(traj, (draw_x     +w_start,     draw_y+h_start), 1, (0,0,255), -1) # learning
	cv2.circle(traj, (draw_x_fast+w_start,draw_y_fast+h_start), 1, (255,0,0), -1) # fast
	cv2.circle(traj, (true_x     +w_start,     true_y+h_start), 1, (0,255,0), -1) # ground truth

	cv2.rectangle(traj, (10, 20), (160, 120), (0,0,0), -1) # erase text from last frame
	text = "Coordinates:"
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	text = "x   = %.2fm"%(x)
	cv2.putText(traj, text, (20,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	text = "y   = %.2fm"%(y)
	cv2.putText(traj, text, (20,80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	text = "z   = %.2fm"%(z)
	cv2.putText(traj, text, (20,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	text = "err = %.2fm"%(error)
	cv2.putText(traj, text, (20,120), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
	

	# cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory Sequence '+sequence_num, traj)
	cv2.waitKey(1)

cv2.imwrite('map_sequence'+sequence_num+'.png', traj)

plt.plot(errors)
plt.ylabel('Error meters')
plt.show()

with open('error_sequence'+sequence_num+'.npy', 'wb') as f:
    np.save(f, errors)