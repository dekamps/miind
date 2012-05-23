#include "AEIFParameter.h"

using namespace PopulistLib;

AEIFParameter::AEIFParameter
(
			Capacity		C_m,
			Conductance		g_l,
			Potential		E_l,
			Potential		V_t,
			Potential		V_r,
			Potential		D_t,
			Time			t_w,
			Conductance		a,
			Current			b
):
_C_m(C_m),
_g_l(g_l),
_E_l(E_l),
_V_t(V_t),
_V_r(V_r),
_D_t(D_t),
_a(a),
_b(b)
{
}

Number AEIFParameter::StateDimension() const
{
	return 2;
}