#include "landing1d.h"

/*********************** Landing1DEnv_cpp ***********************/

/* Class constructor 1
INPUT:
- config = configuration dictionary
*/
Landing1DEnv_cpp::Landing1DEnv_cpp(
    const std::map<std::string,Landing1DEnv_cpp::config_type>& config) : 
    MDPEnv_cpp(-std::numeric_limits<double>::infinity(), 
        0., std::get<int>(config.at("H")), 
        std::get<double>(config.at("tf"))/((double)std::get<int>(config.at("H")))),
    H{std::get<int>(config.at("H"))},
    h0_min{std::get<double>(config.at("h0_min"))},
    h0_max{std::get<double>(config.at("h0_max"))},
    v0_min{std::get<double>(config.at("v0_min"))},
    v0_max{std::get<double>(config.at("v0_max"))},
    m0{std::get<double>(config.at("m0"))},
    tf{std::get<double>(config.at("tf"))},
    hf{std::get<double>(config.at("hf"))},
    vf{std::get<double>(config.at("vf"))},
    Tmax{std::get<double>(config.at("Tmax"))},
    c{std::get<double>(config.at("c"))},
    g{std::get<double>(config.at("g"))},
    dist_h(h0_min, h0_max),
    dist_v(v0_min, v0_max),
    EoM(g, c)
{   
    // Observation space
    observation_space = spaces.attr("Box")("low"_a = -std::numeric_limits<double>::infinity(), 
        "high"_a = std::numeric_limits<double>::infinity(), "shape"_a = py::make_tuple(4), 
        "dtype"_a = py::module_::import("numpy").attr("float64"));

    // Action space
    action_space = spaces.attr("Box")("low"_a = -1., "high"_a = 1., "shape"_a = py::make_tuple(1), 
        "dtype"_a = py::module_::import("numpy").attr("float32"));

    // Seed
    if (config.find("prng_seed") != config.end())
    {
        const std::vector<unsigned int> values = 
            this->seed(std::get<int>(config.at("prng_seed")));
    }
}


/* Get current observation
INPUT:
- state = current state
OUTPUT:
- observation = current observation
*/
std::vector<Landing1DEnv_cpp::obs_type> Landing1DEnv_cpp::get_observation(
    const std::map<std::string,Landing1DEnv_cpp::state_type>& state,
    const std::vector<Landing1DEnv_cpp::control_type>& control)
{
    std::vector<double> observation{
        std::get<double>(state.at("h")),
        std::get<double>(state.at("v")),
        std::get<double>(state.at("m")),
        std::get<double>(state.at("t"))};

    return observation;
}


/* Get current control
INPUT:
- action = current action
OUTPUT:
- control = current control
*/
std::vector<Landing1DEnv_cpp::control_type> 
    Landing1DEnv_cpp::get_control(
        const std::vector<Landing1DEnv_cpp::action_type>& action,
        std::map<std::string,Landing1DEnv_cpp::state_type>& state)
{
    double control = 0.5 * (action[0] + 1.) * Tmax;

    return std::vector<double>(1, control);
}


/* Next system state
INPUT:
- state = current state
- control = current control
OUTPUT:
- next_state = next state
*/
std::map<std::string,Landing1DEnv_cpp::state_type> 
    Landing1DEnv_cpp::next_state(
        const std::map<std::string,Landing1DEnv_cpp::state_type>& state, 
        const std::vector<Landing1DEnv_cpp::control_type>& control,
        const double time_step)
{
    // Current state
    std::vector<double> y{
        std::get<double>(state.at("h")),
        std::get<double>(state.at("v")),
        std::get<double>(state.at("m"))};

    // EoM integration
    std::vector<double> ynew = rk4_method(EoM, y, 
        std::get<double>(state.at("t")), time_step, control[0]);

    // Update state
    std::map<std::string, state_type> next_state;
    next_state["h"] = ynew[0];
    next_state["v"] = ynew[1];
    next_state["m"] = ynew[2];
    next_state["t"] = std::get<double>(state.at("t")) + time_step;
    next_state["step"] = std::get<int>(state.at("step")) + 1;

    return next_state;
}


/* Collect reward
INPUT:
- prev_state = previous state
- state = current state
OUTPUT:
- reward = current reward
- done = is episode done?
*/
void Landing1DEnv_cpp::collect_reward(
    const std::map<std::string,Landing1DEnv_cpp::state_type>& prev_state,
    std::map<std::string,Landing1DEnv_cpp::state_type>& state,
    const std::vector<Landing1DEnv_cpp::control_type>& control,
    double& reward, bool& done)
{
    // Done signal
    done = (std::get<int>(state.at("step")) >= max_episode_steps);
    if (std::get<double>(state.at("h")) <= 0. || 
        std::get<double>(state.at("m")) <= 0.)
        done = true;

    // Reward
    reward = std::get<double>(state.at("m")) - std::get<double>(prev_state.at("m"));
    if (done)
    {
        double cstr_viol = std::max(std::max(fabs(std::get<double>(state.at("h")) - hf), 
            fabs(std::get<double>(state.at("v")) - vf) - 0.005), 0.);
        reward -= 10. * cstr_viol;
        state["cstr_viol"] = cstr_viol;
    }
}


/* Get infos
INPUT:
- prev_state = previous state
- state = current state
- control = last control
- done = is episode done?
OUTPUT:
- info = infos
*/
std::map<std::string,std::map<std::string,std::vector<Landing1DEnv_cpp::info_type>>> 
    Landing1DEnv_cpp::get_info(
        const std::map<std::string,Landing1DEnv_cpp::state_type>& prev_state,
        std::map<std::string,Landing1DEnv_cpp::state_type>& state,
        const std::vector<Landing1DEnv_cpp::obs_type>& observation,
        const std::vector<Landing1DEnv_cpp::control_type>& control,
        const double reward,
        const bool done
    )
{
    std::map<std::string, std::map<std::string, std::vector<double>>> info;

    info["episode_step_data"]["h"] = std::vector<double>{std::get<double>(prev_state.at("h"))};
    info["episode_step_data"]["v"] = std::vector<double>{std::get<double>(prev_state.at("v"))};
    info["episode_step_data"]["m"] = std::vector<double>{std::get<double>(prev_state.at("m"))};
    info["episode_step_data"]["t"] = std::vector<double>{std::get<double>(prev_state.at("t"))};
    info["episode_step_data"]["T"] = std::vector<double>{control[0]};
    if (done)
    {
        info["episode_step_data"]["h"].push_back(std::get<double>(state.at("h")));
        info["episode_step_data"]["v"].push_back(std::get<double>(state.at("v")));
        info["episode_step_data"]["m"].push_back(std::get<double>(state.at("m")));
        info["episode_step_data"]["t"].push_back(std::get<double>(state.at("t")));
        info["episode_step_data"]["T"].push_back(control[0]);
        info["custom_metrics"]["cstr_viol"] = std::vector<double>{std::get<double>(state.at("cstr_viol"))};
    }

    return info;
}


/* Reset
OUTPUT:
- observation: first observation
*/
const std::vector<Landing1DEnv_cpp::obs_type> Landing1DEnv_cpp::reset()
{
    // Reset state
    state["h"] = dist_h(gen);
    state["v"] = dist_v(gen);
    state["m"] = m0;
    state["t"] = 0.;
    state["step"] = 0;

    // Control
    const std::vector<double> control(1, 0.);

    // First observation
    const std::vector<double> observation = this->get_observation(state, control);

    return observation;
}