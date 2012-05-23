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
#include "InitializeAlgorithmGrid.h"
#include "PopulistException.h"

using namespace std;
using namespace PopulistLib;

namespace {

	double gaussian
			(
				Potential potential, 
				const InitialDensityParameter& parameter_density
			)
	{
		if ( parameter_density._sigma == 0 && potential == parameter_density._mu )
			return 1.0;
		if ( parameter_density._sigma == 0 && potential != parameter_density._mu )
			return 0.0;

		// renormalization will occur, prefactor unimportant
		double diff = (parameter_density._mu - potential)/parameter_density._sigma;
		return exp( -0.5*diff*diff );
	}
}

Potential InitializeAlgorithmGrid::DeltaV
(
	Number number_of_initial_bins,
	Potential v_min,
	const PopulationParameter& parameter_population
) const
{
	return (parameter_population._theta - v_min)/(number_of_initial_bins - 1);
}

Index InitializeAlgorithmGrid::IndexReversal
(
	Number number_of_initial_bins,
	Potential v_min,
	const PopulationParameter& parameter_population

	) const
{
	return static_cast<Index>( (parameter_population._V_reversal-v_min)/DeltaV(number_of_initial_bins,v_min,parameter_population) );
}

double InitializeAlgorithmGrid::ExpansionFactorDoubleRebinner
(
	Number number_initial_bins,
	Potential v_min,
	const PopulationParameter& parameter_population
) const
{
	int index_reversal_bin = static_cast<int>( IndexReversal(number_initial_bins,v_min,parameter_population) ); 
	int number_growing_bins = number_initial_bins - index_reversal_bin;

	int number_original_bins = number_growing_bins + index_reversal_bin;
	assert( number_original_bins == number_initial_bins);

	int number_new_bins = 2*number_growing_bins  + index_reversal_bin;

	return static_cast<double>(number_new_bins)/static_cast<double>(number_original_bins);
}

AlgorithmGrid InitializeAlgorithmGrid::InitializeGrid
(
	Number                           number_of_initial_bins,
	Potential                        v_min,
	const PopulationParameter&       parameter_population,
	const InitialDensityParameter&   parameter_density
) const
{
	InitialDensityParameter parameter_initial = parameter_density;

	vector<double> vector_potential (number_of_initial_bins, 0);
	vector<double> vector_state     (number_of_initial_bins, 0);

	// first calculate which bin is closest to V = V_reversal
	double delta_v = 
		DeltaV
		(
			number_of_initial_bins,
			v_min,
			parameter_population
		);

	int index_reversal =
		static_cast<int>(
			IndexReversal
			(
				number_of_initial_bins,
				v_min,
				parameter_population
			)
		);

	// if index_reversal == n initial bins - 1, delta_v is undefined
	if (! (index_reversal >= 0 && index_reversal <= static_cast<int>(number_of_initial_bins) - 1))
		throw PopulistException("Reversal potential is too close to threshold"); 

	vector_potential[index_reversal] = parameter_population._V_reversal;

	// So, we have one bin which exactly matches the reversal potential
	// this is important in determining the zero leak parameters

	// we now recalculate delta_v
	int number_positive_potential_bins = number_of_initial_bins  - index_reversal;
	delta_v = (parameter_population._theta - parameter_population._V_reversal)/(number_positive_potential_bins - 1);


	// realign the potential if sigma == 0, so that one and only one bin corresponds to approximately the desired initial potential
	if (parameter_density._sigma == 0)
	{
		int index_mu_relative_to_reversal = static_cast<int>( (parameter_density._mu - parameter_population._V_reversal)/delta_v );
		parameter_initial._mu = parameter_population._V_reversal  + index_mu_relative_to_reversal*delta_v;
	}

	// fill in the potential bins that are larger than V_reversal
	for
	(
		int index_positive = index_reversal + 1; 
		index_positive < static_cast<int>(number_of_initial_bins); 
		index_positive++ 
	)
		{
			vector_potential[index_positive] = parameter_population._V_reversal + delta_v*(index_positive-index_reversal);

			// renormalize the density later
			vector_state    [index_positive] = 
				gaussian
				(
					vector_potential[index_positive],
					parameter_initial
				);
		}

	// fill in the negative bins
	for
	(
		int index_negative = index_reversal - 1;
		index_negative >= 0;
		index_negative--
	)
		{
			vector_potential[index_negative] = parameter_population._V_reversal - delta_v*(index_reversal - index_negative);

			// renormalize the density later
			vector_state    [index_negative] = 
				gaussian
				(
					vector_potential[index_negative],
					parameter_initial
				);
		}

	// the reversal bin already has correct potential, but the initial value must still be set
		vector_state[index_reversal] = 
			gaussian
			(
					vector_potential[index_reversal], 
					parameter_initial
			);

	double sum = std::accumulate
				(
					vector_state.begin(),
					vector_state.end(),
					0.0
				)/number_of_initial_bins;

	assert( sum > 0 );

	std::transform
		(
			vector_state.begin(),
			vector_state.end(),
			vector_state.begin(),
			bind2nd
			(
				divides<double>(),
				sum
			)
		);


	return AlgorithmGrid
			(
				vector_state,
				vector_potential
			);
}
