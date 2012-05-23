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
#include <cassert>
#include "InitializePotentialVector.h"

using namespace std;
using namespace PopulistLib;

vector<double> PopulistLib::InitializePotentialVector
				(
					Number number_of_initial_bins,
					Potential v_min,
					const PopulationParameter& parameter_population										
				)
{
	vector<double> vector_return(number_of_initial_bins,0);

	// first calculate which bin is closest to V = V_reversal
	double delta_v = (parameter_population._theta - v_min)/(number_of_initial_bins - 1);

	int index_reversal = static_cast<int>( floor(parameter_population._V_reversal-v_min/delta_v) );
	assert(index_reversal >= 0 && index_reversal < static_cast<int>(number_of_initial_bins));

	vector_return[index_reversal] = parameter_population._V_reversal;

	// So, we have one bin which exactly matches the reversal potential
	// this is important in determining the zero leak parameters

	// we now recalculate delta_v
	int number_positive_potential_bins = number_of_initial_bins  - index_reversal;
	delta_v = (parameter_population._theta - parameter_population._V_reversal)/(number_positive_potential_bins - 1);

	// fill in the potential bins that are larger than V_reversal
	for
	(
		int index_positive = index_reversal + 1; 
		index_positive < static_cast<int>(number_of_initial_bins); 
		index_positive++ 
	)
		{
			vector_return[index_positive] = parameter_population._V_reversal + delta_v*(index_positive-index_reversal);
		}

	// fill in the negative bins
	for
	(
		int index_negative = index_reversal - 1;
		index_negative >= 0;
		index_negative--
	)
		{
			vector_return[index_negative] = parameter_population._V_reversal - delta_v*(index_reversal - index_negative);
		}

	return vector_return;
}
