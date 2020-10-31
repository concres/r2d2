import numpy as np 
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from zodbpickle import pickle

sequence_num = "04" # two digits
root_path = "/media/alves/alves32/dataset/"
img_dir    = root_path+"sequences/"+sequence_num+"/image_0/"

detector = cv2.xfeatures2d.SIFT_create()
descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

size_max = float('-inf')
size_min = float('inf')
angle_max = float('-inf')
angle_min = float('inf')
response_max = float('-inf')
response_min = float('inf')
octave_max = float('-inf')
octave_min = float('inf')

num_imgs = 100
# get max values for normalization
for img_id in range(num_imgs+1):
	img_path = img_dir+str(img_id).zfill(6)+".png"
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	kp = detector.detect(img)
	# kp = kp[:1000] # sometimes return more than 1000
	for p in kp:
		if p.size > size_max:
			size_max = p.size
		elif p.size < size_min:
			size_min = p.size

		if p.angle > angle_max:
			angle_max = p.angle
		elif p.angle < angle_min:
			angle_min = p.angle

		if p.response > response_max:
			response_max = p.response
		elif p.response < response_min:
			response_min = p.response

		if p.octave > octave_max:
			octave_max = p.octave
		elif p.octave < octave_min:
			octave_min = p.octave
	print(img_id)


x_train = np.empty((0,6), dtype=np.float32)
y_train = np.zeros(num_imgs*1000, dtype=np.uint8)
img_path = img_dir+'000000.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
kp_ref = detector.detect(img)
kp_ref, des_ref = descriptor.compute(img, kp_ref)
kp_ref = kp_ref[:1000]
des_ref = des_ref[:1000]

for img_id in range(1, num_imgs+1):
	img_path = img_dir+str(img_id).zfill(6)+".png"
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

	kp_cur = detector.detect(img)
	kp_cur, des_cur = descriptor.compute(img, kp_cur)
	kp_cur = kp_cur[:1000]
	des_cur = des_cur[:1000]


	kp = [kp_ref, kp_cur]
	des = [des_ref, des_cur]
	norm = cv2.NORM_HAMMING
	matcher = cv2.BFMatcher(norm)

	# make the matches for one direction
	matches = matcher.knnMatch(des[0],des[1],k=2)
	# Apply ratio test
	good1 = []
	for m,n in matches:
	    if m.distance < 0.55*n.distance:
	        good1.append(m)
	# good = good1

	# make the matches for the other direction
	matches = matcher.knnMatch(des[1],des[0],k=2)
	# Apply ratio test
	good2 = []
	for m,n in matches:
	    if m.distance < 0.55*n.distance:
	        good2.append(m)

	# make the crosscheck
	really_good = []
	for g1 in good1:
		for g2 in good2:
			if g1.queryIdx == g2.trainIdx and g1.trainIdx == g2.queryIdx:
				really_good.append(g1)
				break
	good = really_good

	# normalize values to uint8 (memory)
	for p in kp_ref:
		# n_pt0 = np.uint8(p.pt[0]*255/img.shape[0])
		# n_pt1 = np.uint8(p.pt[1]*255/img.shape[1])
		# n_sz = np.uint8((p.size-size_min)*255/size_max)
		# n_ang = np.uint8((p.angle-angle_min)*255/angle_max)
		# n_resp = np.uint8((p.response-response_min)*255/response_max)
		# n_oct = np.uint8((p.octave-octave_min)*255/octave_max)
		# new_train = np.array([[n_pt0, n_pt1, n_sz, n_ang, n_resp, n_oct]], dtype=np.uint8)
		# x_train = np.append(x_train, new_train, axis=0)
		new_train = np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave]], dtype=np.float32)
		x_train = np.append(x_train, new_train, axis=0)


	# get the label for the good ones
	for g in really_good:
		y_train[((img_id-1)*1000)+m.queryIdx] = 1
	
	kp_ref = kp_cur
	des_ref = des_cur
	print(img_id) # feedback processing velocity


# train classifier
rfc = RandomForestClassifier(n_estimators = 5, max_depth = 5, min_samples_split = 2, n_jobs = -1)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_train)
print('Acurácia treino: %.2f%%' % (accuracy_score(y_train, y_pred)*100))

# save the model to disk
filename = 'rf_model.sav'
pickle.dump(rfc, open(filename, 'wb'))