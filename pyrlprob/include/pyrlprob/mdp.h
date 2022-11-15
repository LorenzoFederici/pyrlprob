#ifndef MDP_H_
#define MDP_H_

#pragma once

#include <random>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <variant>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;


/*  Markov Decision Process in C++ */
template <typename S, 
    typename A, 
    typename C, 
    typename O, 
    typename I,
    typename Co>
class MDPEnv_cpp 
{
    public:
    /* Attributes */
    
    using state_type = S;
    using action_type = A;
    using control_type = C;
    using obs_type = O;
    using info_type = I;
    using config_type = Co;

    py::object spaces, observation_space, action_space;
    std::tuple<double,double> reward_range;
    int max_episode_steps;
    int iter0, iterf;
    double time_step, epsilon0, epsilonf; 
    double epsilon;
    unsigned int prng_seed;
    std::default_random_engine gen;
    std::map<std::string,state_type> prev_state;
    std::map<std::string,state_type> state;
    
    /* Methods */

    /* Class constructor 1
    INPUT:
    - config = configuration dictionary
    */
    MDPEnv_cpp(const double min_reward = -std::numeric_limits<double>::infinity(),
        const double max_reward = std::numeric_limits<double>::infinity(),
        const int max_episode_steps = 999999, const double time_step = 0., 
        const int iter0 = 0, const double epsilon0 = 0.,
        const int iterf = 1, const double epsilonf = 0.) :
        reward_range{std::make_tuple(min_reward, max_reward)}, 
        max_episode_steps{max_episode_steps},
        iter0{iter0}, iterf{iterf}, time_step{time_step}, 
        epsilon0{epsilon0}, epsilonf{epsilonf},
        spaces{spaces = py::module_::import("gym").attr("spaces")}
        {
           const std::vector<unsigned int> values = this->seed();
        };
    
    virtual std::vector<obs_type> get_observation(
        const std::map<std::string,state_type>& state,
        const std::vector<control_type>& control) = 0;

    virtual std::vector<control_type> get_control(
        const std::vector<action_type>& action,
        std::map<std::string,state_type>& state) = 0;

    virtual std::map<std::string,state_type> next_state(
        const std::map<std::string,state_type>& state, 
        const std::vector<control_type>& control,
        const double time_step) = 0;
        
    virtual void collect_reward(const std::map<std::string,state_type>& prev_state,
        std::map<std::string,state_type>& state,
        const std::vector<control_type>& control,
        double& reward, bool& done) = 0;

    virtual std::map<std::string,std::map<std::string,std::vector<info_type>>> 
        get_info(
            const std::map<std::string,state_type>& prev_state,
            std::map<std::string,state_type>& state,
            const std::vector<obs_type>& observation,
            const std::vector<control_type>& control,
            const double reward,
            const bool done
        ) = 0;
    
    /* Set constraint satisfation tolerance
    INPUT:
    - iter = current training iteration
    */
    virtual void set_cstr_tolerance(const int iter)
    {
        if (iter <= iter0)
            epsilon = epsilon0;
        else if (iter >= iterf)
            epsilon = epsilonf;
        else
        {
            epsilon = epsilon0*pow(epsilonf/epsilon0, 
                (double)(iter - iter0)/(double)(iterf - iter0));
        } 
    }
        

    /* Gym Methods */

    /* Seed function
    INPUT:
    - prng_seed_ = a seed for the pseudo-random number generator
    OUTPUT:
    - the given generator
    */
    virtual const std::vector<unsigned int> seed(unsigned int prng_seed_ = time(NULL))
    {
        this->prng_seed = prng_seed_;
        gen.seed(prng_seed);
        srand(prng_seed);
        
        return std::vector<unsigned int>(1, prng_seed);
    }

    /* Step function
    INPUT:
    - action = current action
    OUTPUT:
    - observation, reward, done, info
    */
    virtual const std::tuple<std::vector<obs_type>, double, bool, 
        std::map<std::string,std::map<std::string,std::vector<info_type>>>> 
        step(const std::vector<action_type>& action)
    {
        // Get control
        const std::vector<control_type> control = get_control(action, state);

        // Previous state
        prev_state = state;

        // Next state
        state = next_state(prev_state, control, time_step);

        // Get observation
        const std::vector<obs_type> observation = get_observation(state, control);

        // Get reward and done signal
        double reward;
        bool done;
        collect_reward(prev_state, state, control, reward, done);

        // Compute infos
        std::map<std::string, std::map<std::string, std::vector<info_type>>> 
            info = get_info(prev_state, state, observation, control, reward, done);

        return std::make_tuple(observation, reward, done, info);
    }

    virtual const std::vector<obs_type> reset() = 0;

    virtual void render() {};
};


#endif  // MDP_H_