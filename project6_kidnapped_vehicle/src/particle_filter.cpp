/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;
using std::numeric_limits;

// create only once the default random engine
static default_random_engine gen;

double ParticleFilter::multi_prob(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y)
{
    double gauss_norm = 1.0 / (2 * M_PI * sig_x * sig_y);
    double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) + pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2));
    return gauss_norm * exp(-exponent);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;
  normal_distribution<double> dist_x(x, std[0]); // distribution
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for(auto i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  normal_distribution<double> n_x(0, std_pos[0]);
  normal_distribution<double> n_y(0, std_pos[1]);
  normal_distribution<double> n_theta(0, std_pos[2]);
  for(auto &p : particles)
  {
    double new_x, new_y, new_theta;
    if(yaw_rate == 0)
    {
      new_x = p.x + velocity * delta_t * cos(p.theta);
      new_y = p.y + velocity * delta_t * sin(p.theta);
      new_theta = p.theta;
    }
    else
    {
      double param1 = velocity / yaw_rate;
      double param2 = yaw_rate * delta_t;
      new_x = p.x + param1 * (sin(p.theta + param2) - sin(p.theta));
      new_y = p.y + param1 * (cos(p.theta) - cos(p.theta + param2));
      new_theta = p.theta + param2;
    }
    p.x = new_x + n_x(gen);
    p.y = new_y + n_y(gen);
    p.theta = new_theta + n_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& obs : observations)
  {
    double min_dist = numeric_limits<double>::max();
    for(const auto& p : predicted)
    {
      double dist_temp = dist(obs.x, obs.y, p.x, p.y);
      if(min_dist > dist_temp)
      {
        min_dist = dist_temp;
        obs.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  for(int i = 0; i < num_particles; ++i)
  {
    Particle& p = particles[i];
    double x_p = p.x;
    double y_p = p.y;
    double theta_p = p.theta;

    // 1) find the landmarks that can be reached
    vector<LandmarkObs> predicted_landmarks;
    for(const auto& lm : map_landmarks.landmark_list) //@TODO: can const reference be used here
    {
      if(dist(x_p, y_p, lm.x_f, lm.y_f) <= sensor_range)
      {
        LandmarkObs l_pred;
        l_pred.id = lm.id_i;
        l_pred.x = lm.x_f;
        l_pred.y = lm.y_f;
        predicted_landmarks.push_back(l_pred);
      }
    }

    // 2) transform observations from car-coordinate to map-coordinate
    vector<LandmarkObs> transformed_observations;
    for(const auto& obs : observations)
    {
      LandmarkObs trans_obs_temp;
      double x_m = x_p + obs.x * cos(theta_p) - obs.y * sin(theta_p);
      double y_m = y_p + obs.x * sin(theta_p) + obs.y * cos(theta_p);
      trans_obs_temp.x = x_m;
      trans_obs_temp.y = y_m;
      trans_obs_temp.id = obs.id;
      transformed_observations.push_back(trans_obs_temp);
    }

    // 3) associate observations to the landmarks
    dataAssociation(predicted_landmarks, transformed_observations);

    // 4) update weight
    double particle_weight = 1.0;
    vector<int> associations_vec;
    vector<double> sense_x_vec;
    vector<double> sense_y_vec;

    for(const auto& trans_obs : transformed_observations)
    {
      double x_obs = trans_obs.x;
      double y_obs = trans_obs.y;
      double mu_x, mu_y;
      for(const auto& pred_obs : predicted_landmarks)
      {
        if(trans_obs.id == pred_obs.id)
        {
          mu_x = pred_obs.x;
          mu_y = pred_obs.y;
        }
      }
      particle_weight *= multi_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
      associations_vec.push_back(trans_obs.id);
      sense_x_vec.push_back(x_obs);
      sense_y_vec.push_back(y_obs);
    }
    //cout << " #[D] particle_weight: "  << particle_weight << endl;
    p.weight = particle_weight;
    weights[i] = particle_weight;
    SetAssociations(p, associations_vec, sense_x_vec, sense_y_vec);
  } // particles 
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  discrete_distribution<int> d(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  for(int i = 0; i < num_particles; ++i)
  {
    resampled_particles.push_back(particles[d(gen)]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}