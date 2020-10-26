import numpy as np 
import cv2
# from extract import FeatureExtractor

# STAGE_FIRST_FRAME = 0
# STAGE_SECOND_FRAME = 1
# STAGE_DEFAULT_FRAME = 2
# kMinNumFeature = 1500

# lk_params = dict(winSize  = (21, 21), 
# 				maxLevel = 3,
#              	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# fe = FeatureExtractor()


def filterKeypoints(kp, des, norm):
    bf = cv2.BFMatcher(norm)
    
    # make the matches
    matches = bf.knnMatch(des[0],des[1],k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    src_pts = np.float32([ kp[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    return src_pts, dst_pts


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


class VisualOdometry:
	def __init__(self, cam, annotations, learning=False):
		# self.learning = learning
		# self.frame_stage = 0
		self.first_frame = True
		self.second_frame = False
		self.cam = cam
		self.new_frame = None
		# self.last_frame = nonmaxSuppression
		self.cur_R = None
		self.cur_t = None
		self.kp_ref = None
		self.kp_cur = None
		self.des_ref = None
		self.des_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		with open(annotations) as f:
			self.annotations = f.readlines()
		self.img_path = None
		self.args = None
		
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

	# def processFirstFrame(self):
	# 	self.px_ref = self.detector.detect(self.new_frame)
	# 	self.px_ref, self.des_ref = self.descriptor.compute(self.new_frame, self.px_ref)
	# 	self.frame_stage = STAGE_SECOND_FRAME

	# def processSecondFrame(self):
	# 	if self.feature_based == True:
	# 		self.px_ref = self.detector.detect(self.new_frame)
	# 		self.px_ref, self.des_ref = self.descriptor.compute(self.new_frame, self.px_ref)
	# 		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
	# 	else:
	# 		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
	# 	E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
	# 	_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
	# 	self.frame_stage = STAGE_DEFAULT_FRAME 
	# 	self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		if self.first_frame == True: # get keypoint of the first frame
			self.first_frame = False
			self.second_frame = True
			self.kp_ref = self.detector.detect(self.new_frame)
			self.kp_ref, self.des_ref = self.descriptor.compute(self.new_frame, self.kp_ref)
			return
		# get keypoint of the current frame
		self.kp_cur = self.detector.detect(self.new_frame)
		self.kp_cur, self.des_cur = self.descriptor.compute(self.new_frame, self.kp_cur)
		kp_ref_f, kp_cur_f = filterKeypoints([self.kp_ref, self.kp_cur], [self.des_ref, self.des_cur], cv2.NORM_HAMMING)
		# get first rotation and translation
		if self.second_frame == True:
			self.second_frame = False
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
		# self.last_frame = self.new_frame

# sba = PySBA()
# sba.bundleAdjust()