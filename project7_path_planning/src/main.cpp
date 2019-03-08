#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include <typeinfo>

// for convenience
using nlohmann::json;
using std::cout;
using std::endl;
using std::string;
using std::vector;

int main()
{
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line))
  {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  EgoVehicle ego;
  h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
               &map_waypoints_dx, &map_waypoints_dy, &ego](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                                                                            uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(data);
      if (s != "")
      {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          // Main (ego) car's localization Data
          ego.x = j[1]["x"];
          ego.y = j[1]["y"];
          ego.s = j[1]["s"];
          ego.d = j[1]["d"];
          ego.yaw = j[1]["yaw"];
          ego.speed = j[1]["speed"];
          ego.ref_x = ego.x;
          ego.ref_y = ego.y;
          ego.ref_yaw = deg2rad(ego.yaw);
          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          //cout << "#[I] previous_path_x.size(): " << previous_path_x.size() << endl;
          cout << "#[I] d:" << ego.d << ", lane: " << get_lane(ego.d) << endl;
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];
          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
          json msgJson;
          /*# Our highway path planner #*/
          // step 1: predict ref_v and lane
          int prev_size = previous_path_x.size();
          if (prev_size > 0)
          {
            ego.s = end_path_s;
          }
          vector<OtherVehicle> others;
          for (size_t i = 0; i < sensor_fusion.size(); ++i)
          {
            OtherVehicle other(sensor_fusion[i]);
            others.push_back(other);
          }
          predict(ego, others, prev_size);
          // step 2: create waypoints
          vector<double> ptsx;
          vector<double> ptsy;
          create_waypoints(previous_path_x, previous_path_y, ego,  map_waypoints_s, map_waypoints_x, map_waypoints_y,
                           ptsx, ptsy);
          // step 3: create spline
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          create_spline(previous_path_x, previous_path_y, ptsx, ptsy, ego,
                        next_x_vals, next_y_vals);
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;
          
          auto msg = "42[\"control\"," + msgJson.dump() + "]";
          //cout << "#[I] msg: " << msg << endl;

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        } // end "telemetry" if
      }
      else
      {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    } // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}