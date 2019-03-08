#ifndef HELPERS_H
#define HELPERS_H

#include <math.h>
#include <string>
#include <vector>
#include "vehicle.h"
#include "spline.h"

// for convenience
using std::string;
using std::vector;
using std::cout;
using std::endl;

const double LANE_WIDTH = 4.0;

const double MAX_SPEED = 49.5; // MPH
const double SAFETY_BUFFER= 40.0;    // safety buffer for lane change
const  double VEL_STEP = 0.224; // 0.224
const double WAYPOINTS_SPACE = 30.0;


// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
//   else the empty string "" will be returned.
string hasData(string s)
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos)
  {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos)
  {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

//
// Helper functions related to waypoints and converting from XY to Frenet
//   or vice versa
//

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Calculate distance between two points
double distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Calculate closest waypoint to current x, y position
int ClosestWaypoint(double x, double y, const vector<double> &maps_x,
                    const vector<double> &maps_y)
{
  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for (int i = 0; i < maps_x.size(); ++i)
  {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x, y, map_x, map_y);
    if (dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }
  }

  return closestWaypoint;
}

// Returns next waypoint of the closest waypoint
int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x,
                 const vector<double> &maps_y)
{
  int closestWaypoint = ClosestWaypoint(x, y, maps_x, maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2((map_y - y), (map_x - x));

  double angle = fabs(theta - heading);
  angle = std::min(2 * pi() - angle, angle);

  if (angle > pi() / 2)
  {
    ++closestWaypoint;
    if (closestWaypoint == maps_x.size())
    {
      closestWaypoint = 0;
    }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta,
                         const vector<double> &maps_x,
                         const vector<double> &maps_y)
{
  int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

  int prev_wp;
  prev_wp = next_wp - 1;
  if (next_wp == 0)
  {
    prev_wp = maps_x.size() - 1;
  }

  double n_x = maps_x[next_wp] - maps_x[prev_wp];
  double n_y = maps_y[next_wp] - maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y);
  double proj_x = proj_norm * n_x;
  double proj_y = proj_norm * n_y;

  double frenet_d = distance(x_x, x_y, proj_x, proj_y);

  //see if d value is positive or negative by comparing it to a center point
  double center_x = 1000 - maps_x[prev_wp];
  double center_y = 2000 - maps_y[prev_wp];
  double centerToPos = distance(center_x, center_y, x_x, x_y);
  double centerToRef = distance(center_x, center_y, proj_x, proj_y);

  if (centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for (int i = 0; i < prev_wp; ++i)
  {
    frenet_s += distance(maps_x[i], maps_y[i], maps_x[i + 1], maps_y[i + 1]);
  }

  frenet_s += distance(0, 0, proj_x, proj_y);

  return {frenet_s, frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s,
                     const vector<double> &maps_x,
                     const vector<double> &maps_y)
{
  int prev_wp = -1;

  while (s > maps_s[prev_wp + 1] && (prev_wp < (int)(maps_s.size() - 1)))
  {
    ++prev_wp;
  }

  int wp2 = (prev_wp + 1) % maps_x.size();

  double heading = atan2((maps_y[wp2] - maps_y[prev_wp]),
                         (maps_x[wp2] - maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s - maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp] + seg_s * cos(heading);
  double seg_y = maps_y[prev_wp] + seg_s * sin(heading);

  double perp_heading = heading - pi() / 2;

  double x = seg_x + d * cos(perp_heading);
  double y = seg_y + d * sin(perp_heading);

  return {x, y};
}

int get_lane(double d)
{
  for (int i = 0; i < 3; ++i)
  {
    if ((d > i * LANE_WIDTH) && (d <= (i + 1) * LANE_WIDTH))
      return i;
  }
}

bool check_is_lane_free(const EgoVehicle &ego, const vector<OtherVehicle> others, int direction, int prev_size)
{
  int target_lane = get_lane(ego.d) + direction;
  if (target_lane < 0 || target_lane > 2)
    return false; // false: not safe to change lane to the desired direction or there is no available lane
  bool is_lane_free = true;
  for (const OtherVehicle &other : others)
  {
    if (get_lane(other.d) == target_lane)
    {
      double other_s = other.speed * 0.02 * prev_size + other.s; // future s of other vehicle
      double diff_s = fabs(ego.s - other.s);
      if (diff_s < SAFETY_BUFFER * 0.8) // too close between ego and other (ahead or behind ego)
      {
        // it is dangerous to change lane
        is_lane_free = false;
      }
    }
  }
  return is_lane_free;
}

bool check_is_too_close(const EgoVehicle &ego, const vector<OtherVehicle> others, int prev_size)
{
  bool is_too_close = false;
  int ego_lane = get_lane(ego.d);
  for (const OtherVehicle &other : others)
  {
    if (get_lane(other.d) == ego_lane)
    {
      double other_s = other.speed * 0.02 * prev_size + other.s;
      double diff_s = other_s - ego.s;
      if (diff_s > 0 && diff_s < SAFETY_BUFFER) // in front of ego, within SAFETY_BUFFER there is another vehicle
      {
        is_too_close = true;
        break;
      }
    }
  }
  return is_too_close;
}

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

void create_waypoints(const nlohmann::json &previous_path_x, const nlohmann::json &previous_path_y, EgoVehicle &ego, 
                      const vector<double> &map_waypoints_s, const vector<double> &map_waypoints_x, const vector<double> &map_waypoints_y,
                      vector<double> &ptsx, vector<double> &ptsy)
{
  int prev_size = previous_path_x.size();
  //cout << " #[D] create_waypoint prev_size: " << prev_size << ", ego.x: " << ego.x << ", ego.ref_x: " << ego.ref_x << endl;
  if (prev_size < 2)
  {
    // use two points that make the path tangent to the car
    double prev_car_x = ego.x - cos(ego.ref_yaw); //NOTE: why not use deg2rad
    double prev_car_y = ego.y - sin(ego.ref_yaw);
    ptsx.push_back(prev_car_x);
    ptsx.push_back(ego.x);
    ptsy.push_back(prev_car_y);
    ptsy.push_back(ego.y);
  }
  else
  {
    // redefine reference state as previous path end point
    ego.ref_x = previous_path_x[prev_size - 1];
    ego.ref_y = previous_path_y[prev_size - 1];
    double ref_x_prev = previous_path_x[prev_size - 2];
    double ref_y_prev = previous_path_y[prev_size - 2];
    ego.ref_yaw = atan2(ego.ref_y - ref_y_prev, ego.ref_x - ref_x_prev);
    // use two points that make the path tangent to the previous path endpoint
    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ego.ref_x);
    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ego.ref_y);
  }
  // In Frenet add evenly 30m spaced points ahead of the starting reference
  vector<double> next_wp0 = getXY(ego.s + WAYPOINTS_SPACE, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
  vector<double> next_wp1 = getXY(ego.s + WAYPOINTS_SPACE*2, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
  vector<double> next_wp2 = getXY(ego.s + WAYPOINTS_SPACE*3, (2 + 4 * ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

  ptsx.push_back(next_wp0[0]);
  ptsx.push_back(next_wp1[0]);
  ptsx.push_back(next_wp2[0]);

  ptsy.push_back(next_wp0[1]);
  ptsy.push_back(next_wp1[1]);
  ptsy.push_back(next_wp2[1]);

  // totally 5 points: first two from previous
  for (int i = 0; i < ptsx.size(); ++i)
  {
    // shift car reference angle to 0 degrees
    //@NOTE: how to derive following formula
    double shift_x = ptsx[i] - ego.ref_x;
    double shift_y = ptsy[i] - ego.ref_y;
    double temp_yaw = 0 - ego.ref_yaw;
    // rotation matrix [cos(theta), -sin(theta); sin(theta), cos(theta)]
    ptsx[i] = (shift_x * cos(temp_yaw) - shift_y * sin(temp_yaw));
    ptsy[i] = (shift_x * sin(temp_yaw) + shift_y * cos(temp_yaw));
  }
}

void create_spline(const nlohmann::json &previous_path_x, const nlohmann::json &previous_path_y,
                   const vector<double> &ptsx, const vector<double> & ptsy, const EgoVehicle& ego,
                   vector<double> &next_x_vals, vector<double> &next_y_vals)
{
  // create a spline
  tk::spline s;
  int prev_size = previous_path_x.size();
  // set (x,y) points to the spline
  s.set_points(ptsx, ptsy);

  // define the actual (x,y) points we will use for the planner
  for (int i = 0; i < prev_size; ++i)
  {
    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  // calculate how to break up spline points so that we travel at our desired reference velocity
  double target_y = s(WAYPOINTS_SPACE);
  double target_dist = sqrt(WAYPOINTS_SPACE * WAYPOINTS_SPACE + target_y * target_y);

  double x_add_on = 0;
  // N * 0.02 * vel = d
  // find N  -> find x
  double N = target_dist / (0.02 * ego.ref_vel * 0.447); // mph*0.447 -> m/s

  for (int i = 1; i <= 50 - prev_size; ++i)
  {
    double x_point = x_add_on + (WAYPOINTS_SPACE / N);
    double y_point = s(x_point);
    x_add_on = x_point;
    double x_ref = x_point;
    double y_ref = y_point;
    // rotate back to normal after rotating it earlier
    // from local coordiante -> to global coordinate
    x_point = (x_ref * cos(ego.ref_yaw) - y_ref * sin(ego.ref_yaw));
    y_point = (x_ref * sin(ego.ref_yaw) + y_ref * cos(ego.ref_yaw));
    x_point += ego.ref_x;
    y_point += ego.ref_y;
    next_x_vals.push_back(x_point);
    next_y_vals.push_back(y_point);
  }
}
#endif // HELPERS_H