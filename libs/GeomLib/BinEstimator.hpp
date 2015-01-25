// Copyright (c) 2005 - 2012 Marc de Kamps
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
#ifndef _CODE_LIBS_GEOMLIB_BINESTIMATOR_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_BINESTIMATOR_INCLUDE_GUARD

#include <vector>
#include <gsl/gsl_spline.h>
#include "../UtilLib/UtilLib.h"
#include "OdeParameter.hpp"

using std::vector;
using UtilLib::Index;

namespace GeomLib {

	//! Calculates the coverage corresponding a given bin and a potential difference.

        //! In a non-equidistant grid, it is important to determine which bins correspond
        //! to the probability mass in a given bin. The current bin must be translated by
        //! an amount commensurate to the synaptic efficacy of the connection under consideration.
        //! The translated bin can cover a number of other bins, some partially or it can cover
        //! a single bin partially. The object uses linear search to determine where in which
        //! bin the translated end points of a given bin fall.

        //! TODO: This is a potential cause of inefficiencies. The bin can be found by representing
        //! the inverse function of the characteristics.

	class BinEstimator {
	public:

	        //! Expresses the coverage of a given bin, _index< as a fraction, _alpha.
		struct BinCover {
			int	    _index; //!< Index of the bin under consideration
			double	_alpha; //!< Fraction of the bin that is covered
		};

		//! Type representing begin an end bins of a translated bins and the respective covering fraction.
		typedef std::pair<BinCover,BinCover> CoverPair;

		//! Constructor
		BinEstimator
		(
		 const vector<Potential>&, //! interpretation array of the geometric grid. 
		 const OdeParameter&       //! parameter determining the grid properties.
		);

		~BinEstimator();

		//! Use for testing purposes, for any normal usage CalculateBinCover should be used
		Index SearchBin(Potential) const;

		//! Cover pair looks at the boundaries of the current bins and calculates where they would end up if
		//! a potential difference is applied to them. The translated lower boundary is represented as the
		//! first element of the pair of type BinCover. Its _index member gives the index of the translated boundary
		//! where _alpha gives the fraction of the bin that is covered. The coverage is calculated from the upper boundary
		//! of the corresponding bin in the interpretation array for the first element in the pair, but from the lower 
		//! boundary of the corresponding bin .

		CoverPair 
			CalculateBinCover
			(
				Index i,  //! Index of the bin whose boundaries are shifted.
				Potential //! Difference in potential with respect to the boundaries of bin i. Careful! For an excitatory input this value must be negative
			) const;

		//! Read only access to the interpretation array
		const vector<double>& InterpretationArray() const { return _vec_interpretation; }

	private:

		Potential 
			Translate
			(
				Potential,
				Potential
			) const;

		int 
			SearchBin
			(
				Index, 
				Potential,
				Potential
			) const;

		double 
			BinLowFraction
			(
				Potential, 
				int
			) const;

		double 
			BinHighFraction
			(
				Potential,
				int
			) const;

		const OdeParameter	_par_ode;


		vector<double>		_vec_interpretation;
	};
}
#endif // include guard


