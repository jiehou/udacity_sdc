import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pickle
from collections import deque


# plot two images in one row
def plot_two_images(img1, img2, img1_title="", img2_title="", gray=True):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    if gray:
        ax1.imshow(img1, cmap="gray")
    else:
        ax1.imshow(img1)
    ax1.set_title(img1_title)
    if gray:
        ax2.imshow(img2, cmap="gray")
    else:
        ax2.imshow(img2)
    ax2.set_title(img2_title)


# plot three figure in one row
def plot_three_images(img1, img2, img3, img1_title="", img2_title="", img3_title="", gray=True):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    if gray:
        ax1.imshow(img1, cmap="gray")
    else:
        ax1.imshow(img1)
    ax1.set_title(img1_title)
    if gray:
        ax2.imshow(img2, cmap="gray")
    else:
        ax2.imshow(img2)
    ax2.set_title(img2_title)
    if gray:
        ax3.imshow(img3, cmap="gray")
    else:
        ax3.imshow(img3)
    ax2.set_title(img2_title)


# save an image under the folder named output_images
def save_img(img, name):
    mpimg.imsave(os.path.join("./output_images", name), img)


# Camera calibration #
# takes file names as input
# get objpoints and imgpoints based on calibration images
# draw corners and save images
def calibrate_camera(fnames, flag_draw_corners=False):
    # prepare obj points like(0,0,0), (1,0,0), (2,0,0)...
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # array to store object points and image points from all the images
    obj_points = []  # 3d points in real world space
    img_points = []
    # make a list of calibration images
    count = 1
    plt.figure(figsize=(15, 10))
    for fname in fnames:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chess corners
        ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), True)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            if flag_draw_corners:
                # draw and display the corners
                corners_img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                plt.subplot(5, 4, count)
                plt.axis('off')
                plt.imshow(corners_img)
                img_name = os.path.split(fname)[-1]
                plt.title(img_name)
                output_img_name = 'temp/corners_' + img_name
                cv2.imwrite(output_img_name, corners_img)
                count += 1
    return obj_points, img_points


# it takes an image, object points and image points as input
# compute camera matrix and distortion coefficients
# and save them into a file named mtx_dist.p
def save_calibration_pickle(fname, obj_points, img_points, fname_calibration_pickle, flag_draw=False):
    # (height, width)
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    # print('#[D]: ', img_size)
    # camera matrix, distortion
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    # save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump(dist_pickle, open(fname_calibration_pickle, "wb"))
    if flag_draw:
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title("Original distorted image")
        plt.subplot(2, 1, 2)
        plt.imshow(undist_img)
        plt.title("Undistorted image")
        plt.show()


# load saved calibration parameters
def load_calibration_pickle(fname_calibration_pickle):
    with open(fname_calibration_pickle, "rb") as pickle_file:
        dist_pickle = pickle.load(pickle_file)
        return dist_pickle


# this function takes an image, object points, and image points
# performs the camera calibration, image distortion correction
def undistort_image(img, dist_pickle):
    # (height, width)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img


# compute perspective transform matrix and
# inverse perspective transform matrix and save them as "pickle_data/perspective.p"
def save_perspective_transform_pickle(fname_pespective_pickle):
    # @NOTE: following points used by Udacity
    # src_pts = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    # dst_pts = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    # [top_left, top_right, bottom_right, bottom_left]
    src_pts = np.float32([[595, 460], [725, 460], [1125, 700], [280, 700]])
    dst_pts = np.float32([[250, 0], [1065, 0], [1065, 720], [250, 720]])
    # calculate the perspective transform matrix using src and dst points
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    # save M and Minv using pickle
    perspective_pickle = {"M": M, "Minv": Minv}
    pickle.dump(perspective_pickle, open(fname_pespective_pickle, "wb"))


# load saved perspective pickle data
def load_perspective_transform_pickle(fname_pespective_pickle):
    with open(fname_pespective_pickle, "rb") as pickle_file:
        perspective_pickle = pickle.load(pickle_file)
        return perspective_pickle


# generate warped image
def perspective_transform(img, perspective_pickle):
    img_size = (img.shape[1], img.shape[0])
    M = perspective_pickle["M"]
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_img


# gradients and color spaces #
# absolute threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=15, abs_thresh=(30, 100)):
    # 1) convert to grayscaled image
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) take the derivative
    if orient == 'x':
        sobel_img = cv2.Sobel(grayscaled_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_img = cv2.Sobel(grayscaled_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) compute the absolute value
    abs_sobel = np.absolute(sobel_img)
    # 4) scale to 8-bit
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
    return binary_output


# magnitude threshold
def mag_thresh(img, sobel_kernel=15, mag_thresh=(50, 100)):
    # 1) convert to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) take the derivative
    sobel_x = cv2.Sobel(grayscaled_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(grayscaled_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) compute the magnitude
    mag_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 4) scale to 8-bit
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    # 5) apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


# direction threshold
def dir_threshold(img, sobel_kernel=15, dir_thresh=(0.7, 1.3)):
    # 1) convert to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) take the derivative
    sobel_x = cv2.Sobel(grayscaled_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(grayscaled_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) compute the direction
    sobel_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # 4) apply direction threhold
    binary_output = np.zeros_like(sobel_dir)
    binary_output[(sobel_dir >= dir_thresh[0]) & (sobel_dir <= dir_thresh[1])] = 1
    return binary_output


# color threshold
def col_threshold(img, sc_thresh=(170, 255)):
    # 1) hls
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # 2) apply color threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= sc_thresh[0]) & (s_channel <= sc_thresh[1])] = 1
    return binary_output


# it helps to pick out white and yellow lanes in case of heavy shadow
def yellow_white_mask(img):
    # 1) hls
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) yellow mask
    yellow_lower = np.array([15, 50, 100])
    yellow_upper = np.array([25, 200, 255])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper) // 255
    # 3) white mask
    white_lower = np.array([0, 200, 0])
    white_upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, white_lower, white_upper) // 255
    return cv2.bitwise_or(yellow_mask, white_mask)


# define a function that thresholds the B-channel of LAB
def lab_threshold(img, bc_thresh=(155, 255)):
    # 1) lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]
    # don't normalize if there are no yellows in the image
    if np.max(b_channel) > 175:
        b_channel = b_channel * (255 / np.max(b_channel))
    # 2) apply a threshold
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel >= bc_thresh[0]) & (b_channel <= bc_thresh[1])] = 1
    return  binary_output


# combine sobelx and s-channel thresholds
def combine_threshs_obsolete(img):
    x_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=15, abs_thresh=(30, 100))
    y_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=15, abs_thresh=(30, 100))
    mag_binary = mag_thresh(img, sobel_kernel=15, mag_thresh=(50, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, dir_thresh=(0.7, 1.3))
    col_binary = col_threshold(img, sc_thresh=(170, 255))
    binary_output = np.zeros_like(col_binary)
    binary_output[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1
    return binary_output


# in this version, pay more attention to the yellow and white colors
def combine_threshs(img):
    # yellow / white mask
    color_mask = yellow_white_mask(img)
    # gradient masks
    x_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=9, abs_thresh=(25, 255))
    return cv2.bitwise_or(color_mask, x_binary)


# compute histogram
def compute_histogram(binary_warped_img):
    height = binary_warped_img.shape[0]
    bottom_half = binary_warped_img[height // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


# advanced computer vision #
def find_lane_pixels(binary_warped, draw_window=True, return_img=True):
    img_height = binary_warped.shape[0]
    # take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[img_height // 2:, :], axis=0)  # (1280, )
    # create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # find the peak of the left and right halves of the histogram
    # these will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # hyperparameters
    # choose the number of sliding windows
    nwindows = 9
    # set the width of the window +/- margin
    margin = 100
    # set minimum number of pixels found to recenter window
    minpix = 50
    # set height of windows based on nwindows
    window_height = np.int(img_height // nwindows)
    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = nonzero[0]  # column index
    nonzerox = nonzero[1]  # row index
    # current positions to be updated for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # step through the windows one by one
    for window in range(nwindows):
        # identify window boundaries in x and y
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - window * window_height
        # find the four below boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # draw the windows on the visualization image
        if draw_window:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 3)  # use green color to draw window
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 3)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low)
                          & (nonzerox < win_xleft_high)).nonzero()[0]  # indices
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low)
                           & (nonzerox < win_xright_high)).nonzero()[0]  # indices
        # append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # @NOTE: concatenate the arrays of indices (previouly a list of lists of pixels)
    # len(left_lane_inds) # 9
    # len(left_lane_inds[0]) # 462
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # print(left_lane_inds.shape) # (40369,)
    # print(right_lane_inds.shape)
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if return_img:
        return leftx, lefty, rightx, righty, out_img
    return leftx, lefty, rightx, righty,


# compute fit coefficients based on lane line points
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return (left_fit, right_fit), (left_fitx, right_fitx), ploty


def detect_lines(binary_warped, return_img=False):
    # find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    # fit a second order polynomial to each using 'np.polyfit'
    # print('#[D] len(lefty): ', len(lefty), ", len(leftx): ", len(leftx), ", len(righty): ", len(righty),
    #      ", len(rightx)", len(rightx))
    (left_fit, right_fit), (left_fitx, right_fitx), ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    if return_img:
        # Colors in the left and right lane regions
        # (red, green, blue)
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        # draw left and right lane lines
        for index in range(binary_warped.shape[0]):
            cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
            cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255, 255, 0))
        return (left_fit, right_fit), (left_fitx, right_fitx), ploty, out_img
    return (left_fit, right_fit), (left_fitx, right_fitx), ploty


# compute radiuses of left and right line curvatures in real world
# unit (meter)
def compute_radiuses(line_fits, ploty, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    left_fitx = line_fits[0]
    right_fitx = line_fits[1]
    # fit new polynomials to height(y), width(x) in real world space
    real_left_fit = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    real_right_fit = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # calculate the radius of curvature in real world (unit is meter)
    y_range = np.max(ploty)
    left_rad = ((1 + (2 * real_left_fit[0] * y_range * ym_per_pix + real_left_fit[1]) ** 2) ** 1.5) \
               / np.absolute(2 * real_left_fit[0])
    right_rad = ((1 + (2 * real_right_fit[0]) * y_range * ym_per_pix + real_right_fit[1]) ** 2) ** 1.5 \
                / np.absolute(2 * real_right_fit[0])
    return (left_rad, right_rad)


# computes car's position
# the offset between the middle of car and the middle of the lane
def car_position(img_shape, line_fits, xm_per_pix=3.7 / 700):
    mid_car = img_shape[1] / 2  # assumed car position is at the middle of figure
    left_fitx = line_fits[0]
    right_fitx = line_fits[1]
    mid_lane = (left_fitx[-1] + right_fitx[-1]) / 2  # -1: car is at the bottom of an image
    offset_from_center = (mid_lane - mid_car) * xm_per_pix
    return offset_from_center


# warp detected lane boundaries back onto the original image
def draw_lane(img, line_fits, ploty, Minv):
    height = img.shape[0]
    width = img.shape[1]
    img_size = (width, height)
    # create an image to draw the lines on
    warp_zero = np.zeros((height, width)).astype(np.uint8)
    color_map = np.dstack((warp_zero, warp_zero, warp_zero))
    # recast x any y points into usable format for cv2.fillPoly()
    left_fitx = line_fits[0]
    right_fitx = line_fits[1]
    # @NOTE: magic happens here
    # following lines need to be studied
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # draw the lane onto the warped blank image
    cv2.fillPoly(color_map, np.int_([pts]), (0, 255, 0))
    # warp the blank back to original image
    new_warped_img = cv2.warpPerspective(color_map, Minv, img_size)
    # combine the result
    final_output = cv2.addWeighted(img, 1, new_warped_img, 0.3, 0)
    return final_output


def region_of_interest(img, vertices):
    """
    applies an image mask. only keeps the region of the image defined by the polygon
    formed from `vertices`. the rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pipeline(img):
    fname_calibration_pickle = "pickle_data/mtx_dist.p"
    fname_perspective_pickle = "pickle_data/perspective.p"
    dist_pickle = load_calibration_pickle(fname_calibration_pickle)
    # print("#[D] mtx: ", dist_pickle["mtx"])
    perspective_pickle = load_perspective_transform_pickle(fname_perspective_pickle)
    # print("#[D] M: ", perspective_pickle["M"])
    undistorted_img = undistort_image(img, dist_pickle)
    binary_img = combine_threshs(undistorted_img)  # apply some thresholds
    warped_binary_img = perspective_transform(binary_img, perspective_pickle)
    fit_coeffs, line_fits, ploty = detect_lines(warped_binary_img)
    radiuses = compute_radiuses(line_fits, ploty)
    print('#[D] radiuses: ', radiuses)
    car_pos = car_position(warped_binary_img.shape, line_fits)
    print('#[D] car position: ', car_pos)
    return draw_lane(img, line_fits, ploty, perspective_pickle["Minv"])


# define a class to model line
class Line:
    def __init__(self, buffer_len=16):
        # was the line detected in the last iteration
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.fit = None
        # list of polynomail coefficients of the previous iterantions
        self.last_fits = deque(maxlen=buffer_len)
        # store all detected line pixels
        self.allx = None
        self.ally = None

    def update_line(self, new_fit, detected):
        self.detected = detected
        self.fit = new_fit
        self.last_fits.append(new_fit)

    def average_fit(self):
        return np.mean(self.last_fits, axis=0)


def get_fit_coeffs_by_sliding_window(combined_warped_img, left_line, right_line):
    left_line.allx, left_line.ally, right_line.allx, right_line.ally = \
        find_lane_pixels(combined_warped_img, draw_window=False, return_img=False)
    detected = True
    if not list(left_line.allx) or not list(left_line.ally):
        # do not detect a valid line
        detected = False
        left_fit = left_line.fit
    else:
        left_fit = np.polyfit(left_line.ally, left_line.allx, 2)

    if not list(right_line.allx) or not list(right_line.ally):
        # do not detect a valid line
        detected = False
        right_fit = right_line.fit
    else:
        right_fit = np.polyfit(right_line.ally, right_line.allx, 2)
    left_line.update_line(left_fit, detected)
    right_line.update_line(right_fit, detected)
    return left_line, right_line


def get_fit_coeffs_by_previous_fits(combined_warped_img, left_line, right_line):
    height, width = combined_warped_img.shape
    left_fit = left_line.fit
    right_fit = right_line.fit
    nonzero = combined_warped_img.nonzero()
    nonzero_x = nonzero[1]  # column
    nonzero_y = nonzero[0]  # row
    margin = 100
    tmp_left_x = left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2]
    tmp_right_x = right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2]
    # get left and right lane line indices
    left_lane_inds = (nonzero_x >= tmp_left_x - margin) & (nonzero_x <= tmp_left_x + margin)
    right_lane_inds = (nonzero_x >= tmp_right_x - margin) & (nonzero_x <= tmp_right_x + margin)
    # again, extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]
    detected = True
    # left line
    if not list(leftx) or not list(lefty):
        # do not detect a valid line
        detected = False
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
    # right line
    if not list(rightx) or not list(righty):
        detected = False
    else:
        right_fit = np.polyfit(righty, rightx, 2)
    left_line.update_line(left_fit, detected)
    right_line.update_line(right_fit, detected)
    return left_line, right_line


def generate_line_polynomial(frame_height, left_fit, right_fit):
    ploty = np.linspace(0, frame_height - 1, frame_height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return (left_fitx, right_fitx), ploty


def prepare_output_frame(img_output, img_binary, img_birdeye, frame_num):
    h, w = img_output.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15
    # add a gray rectangle to highlight the upper area
    mask = img_output.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    img_output = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_output, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    img_output[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    img_output[off_y:thumb_h + off_y, 2 * off_x + thumb_w:2 * (off_x + thumb_w), :] = thumb_birdeye

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_output, 'Frame: {}'.format(frame_num), (860, 60), font, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img_output


class ProcessImage:
    def __init__(self, fname_calibration_pickle, fname_pespective_pickle):
        self.dist_pickle = load_calibration_pickle(fname_calibration_pickle)
        self.perspective_pickle = load_perspective_transform_pickle(fname_pespective_pickle)
        self.left_line = Line(buffer_len=16)
        self.right_line = Line(buffer_len=16)
        self.processed_frames = 0
        # region of interest
        self.vertices = np.array([[(120, 710), (120, 0), (1150, 0), (1200, 710)]], dtype=np.int32)

    def __call__(self, img):
        undistorted_img = undistort_image(img, self.dist_pickle)
        binary_img = combine_threshs(undistorted_img)
        warped_binary_img = perspective_transform(binary_img, self.perspective_pickle)
        frame_height = warped_binary_img.shape[0]
        masked_warped_binary_img = region_of_interest(warped_binary_img, self.vertices)
        if (self.processed_frames > 0) and self.left_line.detected and self.right_line.detected:
            self.left_line, self.right_line = get_fit_coeffs_by_previous_fits(masked_warped_binary_img, self.left_line,
                                                                              self.right_line)
        else:
            self.left_line, self.right_line = get_fit_coeffs_by_sliding_window(masked_warped_binary_img, self.left_line,
                                                                               self.right_line)
        left_avg_fit = self.left_line.average_fit()
        right_avg_fit = self.right_line.average_fit()
        avg_line_fits, ploty = generate_line_polynomial(frame_height, left_avg_fit, right_avg_fit)
        final_output = draw_lane(img, avg_line_fits, ploty, self.perspective_pickle["Minv"])
        # compute radiuses
        radiuses = compute_radiuses(avg_line_fits, ploty)
        # compute car offset
        if self.left_line.detected and self.right_line.detected:
            line_fits, ploty = generate_line_polynomial(frame_height, self.left_line.fit, self.right_line.fit)
            car_offset = car_position(warped_binary_img.shape, line_fits)
        else:
            car_offset = -1
        print('#[I] frame: ', self.processed_frames, ', left: ', radiuses[0], ', right: ', radiuses[1], ', offset: ',
              car_offset)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (255, 255, 255)
        cv2.putText(final_output, "Left lane line curvature: {:.0f} m".format(radiuses[0]), (50, 50),
                    font, font_scale, font_color, 2)
        cv2.putText(final_output, "Right lane line curvature: {:.0f} m".format(radiuses[1]), (50, 120),
                    font, font_scale, font_color, 2)
        cv2.putText(final_output, "Car is {:.2f} m right of center".format(car_offset), (50, 190),
                    font, font_scale, font_color, 2)
        cv2.putText(final_output, "Frame: {}".format(self.processed_frames), (50, 260),
                    font, font_scale, font_color, 2)
        # @NOTE: debug
        # final_output = prepare_output_frame(final_output, warped_binary_img, masked_warped_binary_img, self.processed_frames)
        self.processed_frames += 1
        return final_output