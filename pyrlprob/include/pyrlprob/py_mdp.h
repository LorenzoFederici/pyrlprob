#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mdp.h"

template <typename S, 
    typename A, 
    typename C, 
    typename O, 
    typename I,
    typename Co>
class pyMDPEnv_cpp : public MDPEnv_cpp<S,A,C,O,I,Co> 
{
    public:
    /* Inherit the constructors */
    using MDPEnv_cpp<S,A,C,O,I,Co>::MDPEnv_cpp;

    typedef MDPEnv_cpp<S,A,C,O,I,Co> Parent;
    typedef std::map<std::string,S> State;
    typedef std::map<std::string,std::map<std::string,std::vector<I>>> Info;

    /* Trampoline (need one for each virtual function) */
    const std::vector<O> get_observation(
        const std::map<std::string,S>& state) override {
            PYBIND11_OVERRIDE_PURE(
                PYBIND11_TYPE(const std::vector<O>),           /* Return type */
                Parent,      /* Parent class */
                get_observation,               /* Name of function in C++ (must match Python name) */
                state                          /* Argument(s) */
            );
        }
    
    const std::vector<C> get_control(
        const std::vector<A>& action) override {
            PYBIND11_OVERRIDE_PURE(
                PYBIND11_TYPE(const std::vector<C>),           /* Return type */
                Parent,      /* Parent class */
                get_control,               /* Name of function in C++ (must match Python name) */
                action                          /* Argument(s) */
            );
        }
    
    std::map<std::string,S> next_state(
        const std::map<std::string,S>& state, 
        const std::vector<C>& control,
        const double time_step) override {
            PYBIND11_OVERRIDE_PURE(
                State,           /* Return type */
                Parent,      /* Parent class */
                next_state,               /* Name of function in C++ (must match Python name) */
                state,                          /* Argument(s) */
                control,
                time_step
            );
        }
    
    void collect_reward(const std::map<std::string,S>& prev_state,
        std::map<std::string,S>& state,
        const std::vector<C>& control,
        double& reward, bool& done) override {
            PYBIND11_OVERRIDE_PURE(
                void,           /* Return type */
                Parent,      /* Parent class */
                collect_reward,               /* Name of function in C++ (must match Python name) */
                prev_state,                          /* Argument(s) */
                state,
                control,
                reward,
                done
            );
        }
    
    std::map<std::string,std::map<std::string,std::vector<I>>> 
        get_info(
            const std::map<std::string,S>& prev_state,
            std::map<std::string,S>& state,
            const std::vector<C>& control,
            const bool done) override {
            PYBIND11_OVERRIDE_PURE(
                Info,           /* Return type */
                Parent,      /* Parent class */
                get_info,               /* Name of function in C++ (must match Python name) */
                prev_state,                          /* Argument(s) */
                state,
                control,
                done
            );
        }
    
    const std::vector<O> reset() override {
            PYBIND11_OVERRIDE_PURE(
                const std::vector<O>,           /* Return type */
                Parent,      /* Parent class */
                reset               /* Name of function in C++ (must match Python name) */
            );
        }
};