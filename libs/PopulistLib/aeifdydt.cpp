#include "aeifdydt.h"
#include "AEIFParameter.h"

using namespace PopulistLib;

int PopulistLib::aeifdydt
(
	double t, 
	const double y[], 
	double dydt[], 
	void * params
)
{
	AEIFParameter* p_param = (AEIFParameter*)params;
	double C_m = p_param->_C_m;
	double g_l = p_param->_g_l;
	double E_l = p_param->_E_l;
	double V_t = p_param->_V_t;
	double D_t = p_param->_D_t;
	double t_w = p_param->_t_w;
	double a   = p_param->_a;
//	double b   = p_param->_b; not used in 1D

	double tau = g_l/C_m;

	// differential equation for V
	dydt[V] = tau*(-(y[V] - E_l) + D_t*exp((y[V] -V_t)/D_t)) - y[W];
	dydt[W] = (a/t_w)*(y[V] - E_l) - y[W]/t_w;

	return 0;
}