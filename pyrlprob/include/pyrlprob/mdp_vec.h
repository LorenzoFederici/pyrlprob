#pragma once

#include "mdp.h"
#include <omp.h>
#include <unistd.h>

/* Vectorized version of the MDPEnv_cpp env 
  which uses OpenMP to run on multiple CPUs */
template <typename E,
    typename S = typename E::state_type, 
    typename A = typename E::action_type, 
    typename O = typename E::obs_type,
    typename I = typename E::info_type,
    typename Co = typename E::config_type>
class MDPVectorEnv_cpp {

    public:
    /* Attributes */

    using state_type = S;
    using action_type = A;
    using obs_type = O;
    using info_type = I;
    using config_type = Co;
    
    int num_envs, num_threads;
    std::vector<E> envs;

    /* Methods */
    
    /* Class constructor
    INPUT:
    - num_envs = number of parallel environments
    - num_threads = number of openmp threads
    - config = configuration dictionary for each env
    */
    MDPVectorEnv_cpp(const int num_envs, 
        const int num_threads,
        const std::map<std::string,Co>& config) : 
        num_envs{num_envs},
        num_threads{num_threads},
        envs(num_envs, E(config))
    {
        // Set number of omp threads
        omp_set_num_threads(num_threads);

        //Seed the environments
        int i;
        #pragma omp parallel for
        for (i = 0; i < num_envs; i++)
            envs[i].seed(envs[i].prng_seed + i + 1);
    }

    /* Reset the environments
    OUTPUT:
    - obs_vec = vector of observations after reset
    */
    std::vector<std::vector<O>> vector_reset()
    {
        std::vector<std::vector<O>> obs_vec(num_envs);
        int i;

        #pragma omp parallel for shared(obs_vec)
        for (i = 0; i < num_envs; i++)
            obs_vec[i] = envs[i].reset();

        return obs_vec;
    }

    /* Reset a specific environment
    INPUT:
    - index = environment index
    OUTPUT:
    - obs = observation after reset
    */
    const std::vector<O> reset_at(const int index)
    {
        return envs[index].reset();
    }

    /* Step the environments
    INPUT:
    - actions = vector of actions
    OUTPUT:
    - observations, rewards, dones, infos
    */
    std::tuple<std::vector<std::vector<O>>, 
        std::vector<double>, 
        std::vector<bool>, 
        std::vector<std::map<std::string,
        std::map<std::string,std::vector<I>>>>> 
        vector_step(const std::vector<std::vector<A>>& actions)
    {
        std::tuple<std::vector<std::vector<O>>,
                std::vector<double>,
                std::vector<bool>,
                std::vector<std::map<std::string, 
                std::map<std::string,std::vector<I>>>>>
                all_data =
                    std::make_tuple(std::vector<std::vector<O>>(num_envs),
                                    std::vector<double>(num_envs),
                                    std::vector<bool>(num_envs),
                                    std::vector<std::map<std::string, 
                                    std::map<std::string,std::vector<I>>>>(num_envs)
                                    );
        int i;

        #pragma omp parallel for shared(all_data)
        for (i = 0; i < num_envs; i++)
        {
            const std::tuple<std::vector<O>, double, bool,
                std::map<std::string, 
                std::map<std::string, std::vector<I>>>>
                data_batch =
                    envs[i].step(actions[i]);
            std::get<0>(all_data)[i] = std::get<0>(data_batch);
            std::get<1>(all_data)[i] = std::get<1>(data_batch);
            std::get<2>(all_data)[i] = std::get<2>(data_batch);
            std::get<3>(all_data)[i] = std::get<3>(data_batch);
        }

        return all_data;
    }

    /* Get the environments
    OUTPUT:
    - envs = vector of environments
    */
    std::vector<E> get_sub_environments()
    {
        return envs;
    }
};