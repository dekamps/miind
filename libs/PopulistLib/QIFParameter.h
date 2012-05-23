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
#ifndef _CODE_LIBS_POPULISTLIB_QIF_PARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_QIF_PARAMETER_INCLUDE_GUARD

#include "BasicDefinitions.h"
#include "../UtilLib/UtilLib.h"

namespace PopulistLib {

	//! This parameter configures the QIFAlgorithm

	//! The default parameters are chosen such that the normal form of the QIFAlgorithm wil be configured. See the QIFAlgorithm documentation for this

	//! The default constructor will set values, so that the topological normal form will be simulated:
	//! \f[
	//! \frac{dV}{dt} = I + V^{2}
	//! \f]
	//! 
	struct QIFParameter : public Streamable {

		Potential	_V_min;			//! minimum value of membrane potential in simulation
		Potential	_V_peak;		//! value beyond which a neuron will be reset, i.e. also the maximum value of the membrane potential in simulation
		Potential	_V_reset;		//! reset value after spike
		Potential	_I;				//! Although a current, it is best to see this as a shape parameter (see QIFAlgorithm)

		virtual bool ToStream(ostream& ) const;

		virtual bool FromStream(istream&);

		virtual string Tag () const;

	};

	bool operator==(const QIFParameter&, const QIFParameter&);
}

#endif // include guard