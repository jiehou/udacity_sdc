## Project 2: Advanced Lane Finding

---

**Overview**

In this project, I write a program to identify the lane boundaries in a video, which is recorded by a 
front-facing camera on a car.

My pipeline consists of eight steps:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to the center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

All functions and a class used in this project can be found in **helper_functions.py**.

[//]: # (Image References)

[image1]: ./output_images/writeup_undistorted_00.png "Undistorted"
[image2]: ./output_images/writeup_undistorted_01.png "Undistorted"
[image3]: ./output_images/writeup_gradients_and_color_transform.png "CombinedThresholdedBinary"
[image4]: ./output_images/writeup_perspective_transform.png "PerspectiveTransform"
[image5]: ./output_images/writeup_histogram.png "Histogram"
[image6]: ./output_images/writeup_detect_lines.png "DetectLines"
[image7]: ./output_images/writeup_warp_detected_lanes_back.png "WarpBack"

### Step 1: Camera Calibration.
A special set of chessboard images is used for camera calibration. They are stored under "./camera_cal/" folder.
I defined a function named **calibrate_camera(fnames, flag_draw_corners=False)**, which has two input parameters. The first
parameter is the set of file names of recorded camera calibration images, which can be easily got by *glob.glob("./camera_cal/calibration*.jpg")*.
The second parameter decides whether the detected corners for an image will be plotted or not. The main purpose of this function is to generate
*obj_points* and *img_points*, which are used by cv2.calibrateCamera to compute the camera calibration matrix and distortion coefficients.
They are saved as pickle file under "./pickle_data/mtx_dist.p" with help of function named "save_calibration_pickle". Moreover, the saved pickle can
be retrieved using a function named **load_calibration_pickle**.

### Step 2: Apply a distortion correction to raw images.
I defined a function named **undistort_image** to fulfill this job. In this function, cv2.undistort is
used. I applied the distortion correction to two images and obtained following results:
![Undistorted][image1]
![Undistorted][image2]

### Step 3: Use gradients, color transforms, etc., to create a combined thresholded binary image.
In this step, I defined following functions to apply different kinds of thresholds to an image.
* Directional gradient: `abs_sobel_thresh`
* Magnitude gradient: `mag_thresh`
* Direction gradient: `dir_thresh`
* Color threshold: `col_thresh`
* Yellow and white color mask: `yellow_white_mask`

**yellow_white_mask** can be used to pick out the white and yellow lines even in the case of heavy shadows. Moreover, 
a function with name **combine_threshs** is defined to combine some of the listed functions to generate a binary 
thresholded image.

Here is an example of my output for this step.
![CombinedThresholdedBinary][image3]


### Step 4: Apply a perspective transform to change an image's perspective from normal view to top-down view (birds-eye view)
| Source        | Destination   | Position | 
|:-------------:|:-------------:| :-------------:|
| 595, 460      | 250, 0        | left top       |
| 725, 460      | 1065, 0     | right top      |
| 1125, 700     | 1065, 720     | right bottom   |
| 280, 700      | 250, 720      | left bottom    |
Following steps need to be done to perform a perspective transform:
1. Choose source (original image) and destination (warped image) points. Four points are enough for a linear transformation 
from one perspective to another one. The selected source and destination points in my project are listed above.
2. Compute the perspective transform (M) and inverse perspective transform (Minv), and save them under "./pickle_data/perspective.p"
3. M and Minv can be used to warp und unwarp images.
My implemented functions related to perspective transform are as follows:
* Compute M and Minv, and save them in a pickle file: `save_perspective_transform_pickle`
* Load saved M and Minv: `load_perspective_transform_pickle`
* warp an image: `perspective_transform`

Here is an example of transforming perspective of a test image:
![PerspectiveTransform][image4]

### Step 5: Detect lane pixels and fit to find the lane boundary.
The histogram and sliding window techniques are used to detect lane pixels. Below is an example of the 
histogram.

![Histogram][image5]

My function named **find_lane_pixels** use the two highest peask from the histogram as a starting point for determining where the lane
lines are, and then use sliding windows moving upwards in image to find the lane pixels. The hyperparameters related to 
my sliding windows are as follows:
* number of sliding windows: 9
* the width of a window (+/- margin): 100
After the lane pixels have been detected, a function named **fit_poly** is used to find two fitted polynomials based on them.
For the purpose of convenience, I define a function named **detect_lines** to detect lane lines in an image. It is a 
combination  of **find_lane_pixels** and **fit_poly**. 
Here is an example of **detect_lines**:

![DetectLines][image6]

### Step 6: Determine the curvature of the lane and vehicle position with respect to center
```math #yourmathlabel
R = (1 + (2Ay + B)**2)**1.5 / |2A|
```
The above equation is used to compute the radius of the lane line curvature. A function named **compute_radiuses** implements
this equation and returns radiuses for both detected lane lines.
```python
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
```
A function named **car_position** returns the offset between the car and the middle of the used lane.
```python
mid_car = warped_shape[1] / 2  # assumed car position is at the middle of figure
    left_fitx = line_fits[0]
    right_fitx = line_fits[1]
    mid_lane = (left_fitx[-1] + right_fitx[-1]) / 2  # car is at the bottom
    offset_from_center = (mid_lane - mid_car) * xm_per_pix
    return offset_from_center
```

### Step 7: Unwarp the detected lane boundaries back onto the original image.
In this project, a function named **draw_lane** warps the detected lane boundaries back to the original
image, and marks them using green color. Here is an example:
![WarpBack][image7] 

### Step 8: Apply pipeline in a video
I define a class named **ProcessImge** for the video processing. It is a combination of the above
described steps. Moreover, in order to make line detections smooth, we average the previous fit coefficients of detected
lane lines. A class named **Line** is introduced to make the implementation easier.
```python
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
        self.processed_frames += 1
        return final_output
```
The processed video for this project can be found under: ./processed_project_video.mp4.
I also tried my pipeline for challenging videos, they can be processed. However, the lane lines are not identified well.

### Discussion
Through this project, I learned and practiced some knowledge related to camera calibration, perspective transform,
Sobel thresholds, color transform, histogram, and sliding window.
I spent much time in parts of detecting lane lines and video processing. Especially, when an error happens during video
processing, it is not easy to find where is the bug. In my case, I just saved the frame that results in failure, and debug
the save image step by step.
Moreover, my pipeline still needs to be improved, because it does not work well in processing challenging videos.
- Sanity check part is missing in my pipeline. 