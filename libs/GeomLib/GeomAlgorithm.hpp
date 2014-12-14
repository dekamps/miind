// Copyright (c) 2005 - 2014 Marc de Kamps
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

#ifndef _CODE_LIBS_GEOMLIB_GEOMALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_GEOMALGORITHM_INCLUDE_GUARD

#include <boost/circular_buffer.hpp>
#include "../DynamicLib/DynamicLib.h"
#include "AbstractMasterEquation.h"
#include "GeomParameter.h"



namespace GeomLib {

  template <class Weight>
	class GeomAlgorithm : public AbstractAlgorithm<Weight> {
	public:

		typedef typename AbstractAlgorithm<Weight>::predecessor_iterator predecessor_iterator;

		typedef GeomParameter Parameter;

		//! Standard way for user to create algorithm
		GeomAlgorithm
		(
			const GeomParameter&
		);

		//! Copy constructor
		GeomAlgorithm(const GeomAlgorithm&);

		//! virtual destructor
		virtual ~GeomAlgorithm();

		//! An algorithm saves its log messages, and must be able to produce them for a Report
		virtual string LogString() const;

		//! Cloning operation, to provide each DynamicNode with its own 
		//! Algorithm instance
		virtual GeomAlgorithm<Weight>* Clone() const;

		//! A complete serialization of the state of an Algorithm, so that
		//! it can be resumed from its disk representation. NOT IMPLEMENTED.
		virtual bool Dump(ostream&) const ;
		
		//! Configure the algorithm with run time parameter
		virtual bool Configure
		(
			const SimulationRunParameter&
		);

		//! Evolve density over the desired time step
		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time
		);

		//! Current time as maintained by the algorithm
		virtual Time CurrentTime() const;

		//! Current rate of Algorithm
		virtual Rate CurrentRate () const;

		//! state of the algorithm
		virtual NodeState State() const;

		//! Return current AlgorithmGrid
		virtual AlgorithmGrid Grid() const;

		//!
		virtual bool CollectExternalInput
		(
			predecessor_iterator,
			predecessor_iterator
		);

	private:

		bool  IsReportDue() const;

		const	GeomParameter						_par_geom;
		AlgorithmGrid		      					_grid;
		boost::shared_ptr<AbstractOdeSystem>    	_p_system;
	    boost::shared_ptr<AbstractMasterEquation>   _p_zl;

	    bool    _b_zl;
		Time	_t_cur;
	    Time    _t_step;
		Time	_t_report;

		mutable Number	_n_report;
		SpecialBins     _bins; //TODO : is this necessary?
	};
}

#endif // include guard

