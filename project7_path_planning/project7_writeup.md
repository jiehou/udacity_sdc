# CarND-Highway-Planning-Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./images/result_00.png "Result00"
[image2]: ./images/result_01.png "Result01"


## Implementation
Our highway planner is based on the idea presented in the *Q&A* session provided by Udacity. The planner contains three parts:

* Predict: predict the velocity and lane of the ego car for the coming time steps. One time step is 20ms. In case that ego car is close to the front car, we check whether there is a available lane to which the ego car can change its direction safely.
```C++
void predict(EgoVehicle& ego, const vector<OtherVehicle>& others, int time_steps)
{
	bool is_too_close = check_is_too_close(ego, others, time_steps);
	ego.lane = get_lane(ego.d);
	bool is_ready_to_change = false;
	if (is_too_close)
	{
		bool is_left_free = check_is_lane_free(ego, others, -1, time_steps);
		bool is_right_free = check_is_lane_free(ego, others, 1, time_steps);
		if (is_left_free || is_right_free)
		{
			is_ready_to_change = true;
			if (is_left_free)
				ego.lane--;
			else
				ego.lane++;
		}
	}
	cout << "#[I] is_ready_to_change: " << is_ready_to_change << ", is_to_close: " << is_too_close << endl; 
	if (is_ready_to_change || !is_too_close)
	{
		// increase speed
		if (ego.ref_vel < MAX_SPEED)
			ego.ref_vel += VEL_STEP; // MPH
	}
	else {
		ego.ref_vel -= VEL_STEP;
	}
	//cout << "#[I] ref_vel: " << ego.ref_vel << endl;
}
```
* Create waypoints: based on the predicted lane, we create waypoints that will be used to generate a trajectory for the ego car. The space intervals of waypoints are 30, 60 and 90.
```C++
// In Frenet add evenly 30m spaced points ahead of the starting reference
vector<double> next_wp0 = getXY(ego.s + WAYPOINTS_SPACE, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp1 = getXY(ego.s + WAYPOINTS_SPACE*2, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp2 = getXY(ego.s + WAYPOINTS_SPACE*3, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
```
* Create trajectory: based on the created waypoints, the predicted velocity(*ref_vel*) and the reference position(*ref_x, ref_y*), a trajectory can be interpolated by a *C++ spline library(https://github.com/ttk592/spline/)*.

In our implementation, we created two structs named **EgoVehicle** and **OtherVehicle**.
```C++
struct EgoVehicle{
    double s;
    double d;
    double x, y;
    double yaw;
    double speed;
    
    double ref_x;
    double ref_y;
    double ref_yaw; // radian
    double ref_vel; // MPH
    int lane;
    
    EgoVehicle() {
        this->ref_x = 0.0;
        this->ref_y = 0.0;
        this->ref_yaw = 0.0;
        this->ref_vel = 10.0;
        this->lane = 1;
    }
};

struct OtherVehicle {
	double d;
	double vx, vy;
	double speed;
	double s;
	
	OtherVehicle(const nlohmann::json& sensor_fusion) {
		this->vx = sensor_fusion[3];
		this->vy = sensor_fusion[4];
		this->s  = sensor_fusion[5];
		this->d  = sensor_fusion[6];
		this->speed = sqrt(vx * vx + vy * vy); // m/s
	}
};
```

## Experiments
Our planner was tested under the Udacity simulator. We let the ego car run at least 4.32 miles. As can be seen, our planner works well.

![Result00][image1]

![Result01][image2]