#pragma once

#include <vector>

/* Equations of motion of the Landing1D environment */
class Landing1D_EoM
{
    private:
        double g, c;

    public:
        Landing1D_EoM(const double g, const double c) : 
            g(g), c(c) {};

        void operator()(const double time, const std::vector<double>& y, std::vector<double>& ydot, const double T)
        {
            // State
            double v = y[1];
            double m = y[2];

            // Derivatives
            double h_dot = v;
            double v_dot = - g + T/m;
            double m_dot = - T/c;

            ydot.push_back(h_dot);
            ydot.push_back(v_dot);
            ydot.push_back(m_dot);
        } 
};