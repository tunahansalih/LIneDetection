import matplotlib
matplotlib.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

img = plt.imread('roadLane.jpg')
img = np.array(img)
# plt.figure()
# plt.imshow(img)
# points = plt.ginput(n=4, show_clicks=True)
# plt.show()
# print(points)
# plt.close()
# pickle.dump(points, open('points.pickle', 'wb'))

points = pickle.load(open('points.pickle', 'rb'))
print(points)

points = np.flip(np.array(points,dtype=np.float32), axis=1)
print(points)

points = order_points(points)
print(points)
corners = np.array(np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32) * (img.shape[1]-1, img.shape[0]-1), np.float32)
print(corners)
# for x,y in corners.astype(np.int):
#     img[x,y,0] = 255
#     img[x,y,1] = 0
#     img[x,y,2] = 0
# plt.imshow(img)
# plt.show()
# print(points)



M = cv2.getPerspectiveTransform(points, corners)

warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# print(warped)
plt.imshow(warped)
plt.show()