// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _CODE_LIBS_GEOMLIB_CNZLECACHE_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_CNZLECACHE_INCLUDE_GUARD

#include <GeomLib/BinEstimator.hpp>
#include "InputParameterSet.hpp"
#include "SpikingOdeSystem.hpp"

namespace GeomLib {

        //! A BinEstimator object is able to determine the begin and end cover of a given translated bin. When the
        //! synaptic efficacies have not changed, there is no need to recalculate the begin and end points of these
        //! bins, and these values are cached. These cached values, rather than the outcome of a BinEstimator are
        //! used by NumericalMasterEquation.

	template <class Estimator>
	class CNZLCache {
	public:

		//! No default arguments for constructor
		CNZLCache();

		//! Initialize the cache. Must be called before first use.
		void Initialize
		(
			const AbstractOdeSystem&,
			const vector<InputParameterSet>& set,
			const Estimator&
		);

		void InitializeCoverPairs();

		typedef pair<vector<typename Estimator::CoverPair>, vector<typename Estimator::CoverPair> >  input_pair_list;

		const vector<input_pair_list>& List() const {return _vec_coverpair;}

	private:

		bool InputStepsHaveChanged() const;

		Number				       			_n_bins;
		const vector<InputParameterSet>*	_p_vec_set;
		vector<InputParameterSet>	        _old_set;
		const Estimator*	       			_p_estimator;
		const AbstractOdeSystem*		   	_p_sys;

		vector<input_pair_list>  _vec_coverpair;

	};
}

#endif // include guard
