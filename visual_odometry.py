import numpy as np 
import cv2
from extract import FeatureExtractor
fe = FeatureExtractor()
from sklearn.ensemble import RandomForestClassifier
from zodbpickle import pickle

FIRST_FRAME = 0
SECOND_FRAME = 1
DEFAULT_FRAME = 2

class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


# load the model from disk
filename = 'rf_model.sav'
rfc = pickle.load(open(filename, 'rb'))


class VisualOdometry:
	def __init__(self, cam, annotations, learning=False, rf=False):
		self.frame_stage = FIRST_FRAME
		self.learning=learning
		self.cam = cam
		self.new_frame = None
		self.cur_R = None
		self.cur_t = None
		self.kp_ref = None
		self.kp_cur = None
		self.des_ref = None
		self.des_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		# self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.detector = cv2.xfeatures2d.SIFT_create()
		self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		self.rf = rf # use random forest classifier
		self.too_restrictive_count = 0
		
		with open(annotations) as f:
			self.annotations = f.readlines()
		self.img_path = None
		self.args = None

	def matcherKeypoints(self, kp, des, use_rf):
		if self.learning:
			norm = cv2.NORM_L2
		else:
			norm = cv2.NORM_HAMMING
			# norm = cv2.NORM_L2
		if use_rf == True:
			# print("rf_used")
			# filter keypoints with random forest classifier
			# format keypoint for random forest
			
			kp_bkup = kp.copy()
			des_bkup = des.copy()
			kp_rf = np.empty((0,6), dtype=np.float32)
			for p in kp[0]:
				new_train = np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave]], dtype=np.float32)
				kp_rf = np.append(kp_rf, new_train, axis=0)
			# get the good ones
			goods_rf = rfc.predict(kp_rf)

			# kp_old = kp[0].copy()
			# des_old = np.copy(des[0])
			kp[0] = []
			des[0] = np.empty((0,32), dtype=np.uint8)
			for i in range(len(goods_rf)):
				if goods_rf[i] == 1:
					kp[0].append(kp_bkup[0][i])
					des[0] = np.append(des[0], [des_bkup[0][i]], axis=0)

			kp_rf = np.empty((0,6), dtype=np.float32)
			for p in kp[1]:
				new_train = np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave]], dtype=np.float32)
				kp_rf = np.append(kp_rf, new_train, axis=0)
			# get the good ones
			goods_rf = rfc.predict(kp_rf)

			# kp_old = kp[1].copy()
			# des_old = np.copy(des[1])
			kp[1] = []
			des[1] = np.empty((0,32), dtype=np.uint8)
			for i in range(len(goods_rf)):
				if goods_rf[i] == 1:
					kp[1].append(kp_bkup[1][i])
					des[1] = np.append(des[1], [des_bkup[1][i]], axis=0)

		# print(len(kp[0]))
		# print(des[0].shape)
		# print("---")
		# print(len(kp[1]))
		# print(des[1].shape)

		matcher = cv2.BFMatcher(norm)

		# make the matches for one direction
		matches = matcher.knnMatch(des[0],des[1],k=2)
		# Apply ratio test
		good1 = []
		# print(len(matches))
		for m,n in matches:
		    if m.distance < 0.85*n.distance:
		        good1.append(m)

		# print(len(good1))

		# make the matches for the other direction
		matches = matcher.knnMatch(des[1],des[0],k=2)
		# Apply ratio test
		good2 = []
		# print(len(matches))

		for m,n in matches:
		    if m.distance < 0.85*n.distance:
		        good2.append(m)

		# print(len(good2))

		# make the crosscheck
		really_good = []
		for g1 in good1:
			for g2 in good2:
				if g1.queryIdx == g2.trainIdx and g1.trainIdx == g2.queryIdx:
					really_good.append(g1)
					break
		good = really_good

		# print(len(good))
		# matcher = cv2.BFMatcher(norm, crossCheck=True)
		# good = matcher.matcher(des[0], des[1])

		if self.learning:
			src_pts = np.float32([ kp[0][m.queryIdx] for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp[1][m.trainIdx] for m in good ]).reshape(-1,1,2)
		else:
			src_pts = np.float32([ kp[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if use_rf == True and (len(src_pts) < 9 or len(dst_pts) < 9):
			# print("kkk eae man")
			# print(len(kp_bkup[0]))
			# print(des_bkup[0].shape)
			src_pts, dst_pts = self.matcherKeypoints(kp_bkup, des_bkup, False)
			self.too_restrictive_count += 1

		return src_pts, dst_pts

	def getKeypointsAndDescriptors(self, img):
		if self.learning:
			kp, des = fe.extract_keypoints(self.img_path)
			# kp = kp[:1000]
			# des = kp[:1000]
		else:
			kp = self.detector.detect(img)
			kp, des = self.descriptor.compute(img, kp)
		return kp, des

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def processFrame(self, frame_id):
		if self.frame_stage == FIRST_FRAME: # get keypoint of the first frame
			self.frame_stage = SECOND_FRAME
			self.kp_ref, self.des_ref = self.getKeypointsAndDescriptors(self.new_frame)
			return
		# get keypoint of the current frame
		self.kp_cur, self.des_cur = self.getKeypointsAndDescriptors(self.new_frame)
		kp_ref_f, kp_cur_f = self.matcherKeypoints([self.kp_ref, self.kp_cur], [self.des_ref, self.des_cur], self.rf)
		# print(len(kp_ref_f))
		# print(len(kp_cur_f))
		# get first rotation and translation
		if self.frame_stage == SECOND_FRAME:
			self.frame_stage = DEFAULT_FRAME
			E, mask = cv2.findEssentialMat(kp_cur_f, kp_ref_f, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
			_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, kp_cur_f, kp_ref_f, focal=self.focal, pp = self.pp)
			return
		E, mask = cv2.findEssentialMat(kp_cur_f, kp_ref_f, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, kp_cur_f, kp_ref_f, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id)
		if(absolute_scale > 0.1):
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
			self.cur_R = R.dot(self.cur_R)
		# pass current to reference
		self.kp_ref = self.kp_cur
		self.des_ref = self.des_cur

	def update(self, img, frame_id, img_path):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		self.img_path = img_path
		self.processFrame(frame_id)