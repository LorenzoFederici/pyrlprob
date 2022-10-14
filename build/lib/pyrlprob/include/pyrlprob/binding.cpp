#include <string>
#include "py_mdp.h"

template <typename S, 
    typename A, 
    typename C, 
    typename O, 
    typename I,
    typename Co,
    class E>
void declare_mdp_class_type(py::module &m, const char* env_name)
{
    py::class_<MDPEnv_cpp<S,A,C,O,I,Co>, pyMDPEnv_cpp<S,A,C,O,I,Co>>(m, "MDPEnv_cpp")
        .def(py::init<const double, const double, const int, const double,
            const int, const double, const int, const double&>(), 
            py::arg("min_reward") = -std::numeric_limits<double>::infinity(), 
            py::arg("max_reward") = std::numeric_limits<double>::infinity(),
            py::arg("max_episode_steps") = 999999,
            py::arg("time_step") = 0.,
            py::arg("iter0") = 0,
            py::arg("epsilon0") = 0.,
            py::arg("iterf") = 1,
            py::arg("epsilonf") = 0.)
        .def_readwrite("observation_space", &MDPEnv_cpp<S,A,C,O,I,Co>::observation_space)
        .def_readwrite("action_space", &MDPEnv_cpp<S,A,C,O,I,Co>::action_space)
        .def_readwrite("reward_range", &MDPEnv_cpp<S,A,C,O,I,Co>::reward_range)
        .def_readwrite("max_episode_steps", &MDPEnv_cpp<S,A,C,O,I,Co>::max_episode_steps)
        .def("get_observation", &MDPEnv_cpp<S,A,C,O,I,Co>::get_observation)
        .def("get_control", &MDPEnv_cpp<S,A,C,O,I,Co>::get_control)
        .def("next_state", &MDPEnv_cpp<S,A,C,O,I,Co>::next_state)
        .def("collect_reward", &MDPEnv_cpp<S,A,C,O,I,Co>::collect_reward)
        .def("get_info", &MDPEnv_cpp<S,A,C,O,I,Co>::get_info)
        .def("set_cstr_tolerance", &MDPEnv_cpp<S,A,C,O,I,Co>::set_cstr_tolerance)
        .def("seed", &MDPEnv_cpp<S,A,C,O,I,Co>::seed, py::arg("prng_seed") = time(NULL))
        .def("step", &MDPEnv_cpp<S,A,C,O,I,Co>::step)
        .def("reset", &MDPEnv_cpp<S,A,C,O,I,Co>::reset)
        .def("render", &MDPEnv_cpp<S,A,C,O,I,Co>::render);

    py::class_<E, MDPEnv_cpp<S,A,C,O,I,Co>>(m, env_name)
        .def(py::init<const std::map<std::string,Co>&>());
}

template <typename E,
    class V,
    typename S = typename E::state_type, 
    typename A = typename E::action_type, 
    typename O = typename E::obs_type,
    typename I = typename E::info_type,
    typename Co = typename E::config_type>
void declare_mdp_vector_class_type(py::module &m, const char* env_name)
{
    py::class_<MDPVectorEnv_cpp<E>>(m, "MDPVectorEnv_cpp")
        .def(py::init<const int, const int,
            const std::map<std::string,Co>&>())
        .def_readwrite("envs", &MDPVectorEnv_cpp<E>::envs)
        .def("vector_reset", &MDPVectorEnv_cpp<E>::vector_reset)
        .def("reset_at", &MDPVectorEnv_cpp<E>::reset_at)
        .def("vector_step", &MDPVectorEnv_cpp<E>::vector_step)
        .def("get_sub_environments", &MDPVectorEnv_cpp<E>::get_sub_environments);
        
    py::class_<V, MDPVectorEnv_cpp<E>>(m, env_name)
        .def(py::init<const int, const int,
            const std::map<std::string,Co>&>());
}