import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle
from project2.binary_thresholder import *
from project2.helper_functions import *
from moviepy.editor import VideoFileClip

fname_calibration_pickle = "pickle_data/mtx_dist.p"
fname_perspective_pickle = "pickle_data/perspective.p"

"""
# save_perspective_transform_pickle(fname_perspective_pickle)
perspective_pickle = load_perspective_transform_pickle(fname_perspective_pickle)
img = mpimg.imread("test_images/test1.jpg")
warped_img = perspective_transform(img, perspective_pickle)
binary_warped = combine_threshs(warped_img)
(left_fit, right_fit), (left_fitx, right_fitx), ploty, lines_img = detect_lines(binary_warped, True)
plot_two_images(binary_warped, lines_img, img1_title="combined binary warped", img2_title="detected lines", gray=True)
"""

# pipeline test
"""
img = mpimg.imread("test_images/test1.jpg")
print(img.shape)
final_output = pipeline(img)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Orignial Image', fontsize=30)
ax2.imshow(final_output)
ax2.set_title('Final Image', fontsize=30)
"""


# video processing
"""
prcoess_image = ProcessImage(fname_calibration_pickle, fname_perspective_pickle)
print('#[I] M: ', prcoess_image.perspective_pickle['M'])
my_clip = VideoFileClip("project_video.mp4")
clip  = my_clip.fl_image(prcoess_image)
clip.write_videofile("processed_project_video.mp4", audio=False)
print("#[I] processed frames: ", prcoess_image.processed_frames)
"""

"""
img1 = mpimg.imread("test_images/test1.jpg")
img2 = mpimg.imread("test_images/test2.jpg")
ret = cv2.matchShapes(img1[:, :, 0], img2[:, :, 0], 1, 0.0)
print(ret)
"""

"""
img = cv2.imread("./test_images/test1.jpg")
dist_pickle = load_calibration_pickle(fname_calibration_pickle)
perspective_pickel = load_perspective_transform_pickle(fname_perspective_pickle)
print(perspective_pickel["M"])
print(perspective_pickel["Minv"])
undistorted_img = undistort_image(img, dist_pickle)
warped_img = perspective_transform(undistorted_img, perspective_pickel)
binary_warped_img = combine_threshs(warped_img)
histogram = compute_histogram(binary_warped_img)
plt.title('Histogram', fontsize=14)
plt.xlabel('Pixel position')
plt.ylabel('Counts')
plt.plot(histogram)
"""

"""
save_perspective_transform_pickle(fname_perspective_pickle)
perspective_pickel = load_perspective_transform_pickle(fname_perspective_pickle)
print(perspective_pickel["M"])
print(perspective_pickel["Minv"])
"""

"""
test1 = cv2.imread("./test_images/test1.jpg")
binary_test1 = combine_threshs(test5)
plot_two_images(out1_test1, out2_test5, "method1", "method2", True)
plt.show()
"""