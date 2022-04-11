#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "landing1d.cpp"

namespace py = pybind11;

PYBIND11_MODULE(landing1d_cpp, m) {
    py::class_<Landing1DEnv_cpp>(m, "Landing1DEnv_cpp")
        .def(py::init<const std::map<std::string,std::variant<int,double>>&>())
        .def("get_observation", &Landing1DEnv_cpp::get_observation)
        .def("get_control", &Landing1DEnv_cpp::get_control)
        .def("next_state", &Landing1DEnv_cpp::next_state)
        .def("collect_reward", &Landing1DEnv_cpp::collect_reward)
        .def("get_info", &Landing1DEnv_cpp::get_info)
        .def("seed", &Landing1DEnv_cpp::seed, py::arg("prng_seed") = time(NULL))
        .def("step", &Landing1DEnv_cpp::step)
        .def("reset", &Landing1DEnv_cpp::reset)
        .def("render", &Landing1DEnv_cpp::render);

    py::class_<Landing1DVectorEnv_cpp>(m, "Landing1DVectorEnv_cpp")
        .def(py::init<const int, const int,
            const std::map<std::string,std::variant<int,double>>&>())
        .def("vector_reset", &Landing1DVectorEnv_cpp::vector_reset)
        .def("reset_at", &Landing1DVectorEnv_cpp::reset_at)
        .def("vector_step", &Landing1DVectorEnv_cpp::vector_step)
        .def("get_sub_environments", &Landing1DVectorEnv_cpp::get_sub_environments);
}
