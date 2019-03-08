#ifndef VEHICLE_H
#define VEHICLE_H
#include <vector>
#include <math.h>
#include "json.hpp"

using std::vector;

struct OtherVehicle {
	double d;
	double vx, vy;
	double speed;
	double s;
	
	//OtherVehicle() { }
	OtherVehicle(const nlohmann::json& sensor_fusion) {
		this->vx = sensor_fusion[3];
		this->vy = sensor_fusion[4];
		this->s  = sensor_fusion[5];
		this->d  = sensor_fusion[6];
		this->speed = sqrt(vx * vx + vy * vy); // m/s
	}
};

struct EgoVehicle{
	double s;
	double d;
	double x, y;
	double yaw;
	double speed;
  
	double ref_x;
	double ref_y;
	double ref_yaw;
  	double ref_vel;
  	int lane;
  
	EgoVehicle() 
    {
      this->ref_x = 0.0;
      this->ref_y = 0.0;
      this->ref_yaw = 0.0;
      this->ref_vel = 10.0;
      this->lane = 1;
    }
};
#endif