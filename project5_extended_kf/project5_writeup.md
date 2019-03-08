# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)

[image1]: ./output_images/project_structure.png "ProjectStructure"
[image2]: ./output_images/dataset1_test.png "Dataset1Test"
[image3]: ./output_images/dataset2_test.png "Dataset2Test"


In this project we will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases).

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets). Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

The sturcture of this project is displayed in the following figure:
![Project structure][image1]

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
4. Run it: `./ExtendedKF `

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project resources page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/382ebfd6-1d55-4487-84a5-b6a5a4ba1e47)
for instructions and the project rubric.

## Files in the src folder
* main.cpp
    - it builds connection to the Udacity simulator
    - it receives data from simulator and passes it to the object **fusionEKF**
    - it computes the RMSE with help of object **tools** 
    - it sends the compute RMSE back to the simulator.
* FusionEKF.cpp
    - it initialize the extended Kalman filter (**ekf_**)
    - it calls the Predict and Update functions of **ekf_**
* kalman_filter.cpp
    - it is the class of extended Kalman filter
    - it has a **Predict** function
    - it has different Update functions for laser and radar measurements: **Update** and **UpdateEKF**
* tools.cpp
    - it calculates the Jacobian matrix for radar case
    - it calculates the RMSE

## Results
Our program was tested using the two provided datasets.
The result from Dataset1 is displayed in the following figure:
![Dataset1 result][image2]

As can be seen, the RMSE of Dataset1 is: [0.0973, 0.0855, 0.4513, 0.4399].

Similarly, the result from Dataset2 is as follows:
![Dataset2 result][image3]

Its RMSE is: [0.0726, 0.0965, 0.4216, 0.4932]