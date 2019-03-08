## Reflection

### 1 Describe the effect of the P,I,and D components
* P stands for "proportional" gain. It enables the steer in proportional to the cross-track error (CTE). In this project, CTE is how far between the car and the middle line of the road. If the proportional gain is too large, the system can become unstable. In contrast, a small gain can make the system less responsive to a large input error.

* I denotes "integral" gain. The integral gain contributes to both the magnitude of the error and the duration of the error. Therefore, it helps reducing residual errors which cannot be eliminated by the P or D component.

* D means "derivative" gain. It takes into account the derivative of the error and helps effectively reducing the oscillations.

### 2 Describe how the final hyperparameters were chosen?
The link (https://robotics.stackexchange.com/questions/167/what-are-good-strategies-for-tuning-pid-loops) gives me some hints about how to tune a PID. I tune my PID with following steps:

1. Set all gains to zero
2. Increase the P gain until the car is able to follow the curves and starts to oscillate. We started kp with a value of 0.01, and it settled in 0.12. The incremental step is 0.01. 
3. Increase the D gain until the car stops oscillating. It settled in 1.1. 1.0 is also acceptable. 
4. After kp and kd were found, we tried to tune the I component in order to enable the car drive smoothly in the sharp curves. Since the integral of the error is a large value, the I should be a smaller value. We started it with 0.0001 and stopped at 0.0005. Because at 0.0005, the car starts oscillating again. From our point of view, the I does not help much. Therefore, we set it as zero.

Finally, our manual tuned parameter is: kp=0.1, ki=0.0, kd=1.1