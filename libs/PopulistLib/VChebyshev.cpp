// Copyright (c) 2005 - 2009 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "VChebyshev.h"
#include "LocalDefinitions.h"
#include "VArray.h"

using namespace PopulistLib;

VChebyshev::VChebyshev():
_mat_p_series(N_CIRC_MAX_CHEB, N_NON_CIRC_MAX_CHEB)
{
	for (Index k = 0; k < N_CIRC_MAX_CHEB; k++)
		for (Index j = 0; j < N_NON_CIRC_MAX_CHEB; j++ )
			_mat_p_series(k, j) = GenerateSeries(k,j);
}

VChebyshev::~VChebyshev()
{
	for (Index k = 0; k < N_CIRC_MAX_CHEB; k++)
		for (Index j = 0; j < N_NON_CIRC_MAX_CHEB; j++ )
			gsl_cheb_free(_mat_p_series(k, j));
}

namespace {

	struct VkjParameters
	{
		int _k;
		int _j;
	};

	double f( double tau, void* p_param)
	{
		VkjParameters* p_params = static_cast<VkjParameters*>(p_param);

		VArray v_array;
		v_array.FillArray(N_CIRC_MAX_CHEB, N_NON_CIRC_MAX_CHEB, tau);

		double v_ret = v_array.V(p_params->_k, p_params->_j);

		return v_ret;
	}
}

gsl_cheb_series* VChebyshev::GenerateSeries
(
	Index k, 
	Index j
)
{
	VkjParameters param = {k, j};

	gsl_function F;
     
	F.function = f;
	F.params   = static_cast<void*>(&param);
	
	
	int n_order = 1;
	gsl_cheb_series* p_series;

	do 
	{
		if ( n_order > 1 )

		gsl_cheb_free(p_series);

		p_series = gsl_cheb_alloc (n_order);

		n_order++;
		gsl_cheb_init
		(
			p_series,
			&F, 
			T_CHEB_MIN,
			T_CHEB_MAX
		);
	}
	while ( fabs( p_series->c[n_order - 1] ) > CHEB_PRECISION );	

	return p_series;
}

