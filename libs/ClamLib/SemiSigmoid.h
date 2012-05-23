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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_CLAMLIB_SEMISIGMOID_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_SEMISIGMOID_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"

using DynamicLib::AlgorithmGrid;
using DynamicLib::NodeState;
using DynamicLib::D_AbstractAlgorithm;
using DynamicLib::Rate;
using DynamicLib::SimulationRunParameter;
using DynamicLib::Time;
using DynamicLib::WilsonCowanParameter;

namespace ClamLib {

	class SemiSigmoid : public D_AbstractAlgorithm {
	public:

		SemiSigmoid(istream& );

		SemiSigmoid(const WilsonCowanParameter&);

		//! copy constructor
		SemiSigmoid(const SemiSigmoid&);

		//! copy operator
		SemiSigmoid& operator=(const SemiSigmoid&);


		virtual ~SemiSigmoid();

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			DynamicLib::Time 
		);

		//! clone
		virtual SemiSigmoid* Clone() const;

		//! report initial state
		virtual AlgorithmGrid Grid() const;

		//! report node state
		virtual NodeState State() const;

		//!
		virtual string LogString() const {return string("");}

		//! streaming to output stream
		virtual bool ToStream(ostream&) const;

		//! streaming from input stream
		virtual bool FromStream(istream&);

		//! stream Tag
		string Tag() const;

		virtual bool Dump(ostream&) const;

		virtual bool Configure
		(
			const SimulationRunParameter&
		);

		virtual Time CurrentTime() const; 

		virtual Rate CurrentRate() const;

		WilsonCowanParameter Parameter() const;
	private:

		void	StripHeader			(istream&);
		void	StripFooter			(istream&);

		WilsonCowanParameter WilsonCowanParameterFromStream(istream&);

		vector<double> InitialState() const;
		WilsonCowanParameter				_parameter;
		DVIntegrator<WilsonCowanParameter>	_integrator;

	};
}

#endif // include guard
