#include <pyrlprob/binding.cpp>
#include "landing1d.cpp"

namespace py = pybind11;

PYBIND11_MODULE(landing1d_cpp, m) {
    declare_mdp_class_type<std::variant<int,double>, 
        double, double, double, double, std::variant<int,double>, Landing1DEnv_cpp>(m, "Landing1DEnv_cpp");
    declare_mdp_vector_class_type<Landing1DEnv_cpp, Landing1DVectorEnv_cpp>(m, "Landing1DVectorEnv_cpp");
}
