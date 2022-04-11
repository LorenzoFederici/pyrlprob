#ifndef RK4
#define RK4

#pragma once

#include <vector>

/* Runge-Kutta 4th order 
   numerical integration method - vector<double>
INPUT:
- f = first order derivatives
- y0 = value of the state variables at t0
- t0 = initial time
- dt = time step
- args = other parameters to pass to function f
OUTPUT:
- y = value of the state variables at t0 + dt
*/
template<typename Func, typename... Args>
std::vector<double> rk4_method(Func& f, const std::vector<double>& y0, const double t0, const double dt, const Args... args)
{
    int n_var = y0.size();

    std::vector<double> f0;
    f(t0, y0, f0, args...);

	const double t1 = t0 + dt / 2.0;
	std::vector<double> y1;
    for (int i=0; i<n_var; i++)
    {
        y1.push_back(y0[i] + dt * f0[i] / 2.0);
    }
    std::vector<double> f1;
    f(t1, y1, f1, args...);

	const double t2 = t0 + dt / 2.0;
	std::vector<double> y2;
    for (int i=0; i<n_var; i++)
    {
        y2.push_back(y0[i] + dt * f1[i] / 2.0);
    }
    std::vector<double> f2;
    f(t2, y2, f2, args...);

	const double t3 = t0 + dt;	
	std::vector<double> y3;
    for (int i=0; i<n_var; i++)
    {
        y3.push_back(y0[i] + dt * f2[i]);
    }
    std::vector<double> f3;
    f(t3, y3, f3, args...);
	
	std::vector<double> y;
    for (int i=0; i<n_var; i++)
    {
        y.push_back(y0[i] + dt * (f0[i] + 2.0 * f1[i] + 2.0 * f2[i] + f3[i]) / 6.0);
    }
        
	return y;
};

#endif