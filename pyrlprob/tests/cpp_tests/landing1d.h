#ifndef LANDING1D_H_
#define LANDING1D_H_

#pragma once

#include <pyrlprob/mdp.h>
#include <pyrlprob/mdp_vec.h>

#include "landing1d_dyn.h"
#include "rk4.h"

/* Landing 1D RL Environment */
class Landing1DEnv_cpp : 
  public MDPEnv_cpp<std::variant<int,double>, 
    double, double, double, double, std::variant<int,double>>
{
  /* Attributes */
  public:

  int H;
  double h0_min, h0_max, v0_min, v0_max, m0, 
    tf, hf, vf, Tmax, c, g;
  std::uniform_real_distribution<double> dist_h, dist_v;
  Landing1D_EoM EoM;

  /* Methods */

  Landing1DEnv_cpp(const std::map<std::string,config_type>& config);
  std::vector<obs_type> get_observation(
    const std::map<std::string,state_type>& state,
    const std::vector<control_type>& control);
  std::vector<control_type> get_control(
    const std::vector<action_type>& action,
    std::map<std::string,state_type>& state);
  std::map<std::string,state_type> next_state(
    const std::map<std::string,state_type>& state, 
    const std::vector<control_type>& control,
    const double time_step);
  void collect_reward(const std::map<std::string,state_type>& prev_state,
      std::map<std::string,state_type>& state,
      const std::vector<control_type>& control,
      double& reward, bool& done);
  std::map<std::string,std::map<std::string,std::vector<info_type>>> 
    get_info(
      const std::map<std::string,state_type>& prev_state,
      std::map<std::string,state_type>& state,
      const std::vector<obs_type>& observation,
      const std::vector<control_type>& control,
      const double reward,
      const bool done
    );
  
  /* Gym Methods */
  const std::vector<obs_type> reset();
};


/* Vectorized version of the Landing1DEnv_cpp env 
  which uses OpenMP to run on multiple CPUs */
class Landing1DVectorEnv_cpp : 
  public MDPVectorEnv_cpp<Landing1DEnv_cpp> 
{
  public:

  Landing1DVectorEnv_cpp(const int num_envs, 
    const int num_threads,
    const std::map<std::string,config_type>& config) :
    MDPVectorEnv_cpp(num_envs, num_threads, config) {};
};



#endif  // LANDING1D_H_