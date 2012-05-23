// Copyright (c) 2005 - 2008 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_ABSTARCTSQUASHINGPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_ABSTARCTSQUASHINGPARAMETER_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"

using UtilLib::Index;
using UtilLib::Number;
using UtilLib::Streamable;

namespace NetLib {

	//! AbstractSquashingParameter

	//! A designed of a concrete squashing function may need a concrete
	//! set of squashingparameters. This class sereves to enable virtual
	//! construction of derived squashing parameter classes.

	class AbstractSquashingParameter : public Streamable
	{
	public:

		virtual ~AbstractSquashingParameter() = 0;

		virtual bool   FromStream(istream&)       = 0;
		virtual bool   ToStream  (ostream&) const = 0;

		bool	InsertParameter	(Index, double);
		double	GetParameter	(Index) const;

		Number NumberOfParameters() const;

	protected:

		//! Future SquashingParameters can have a number of parameters that
		//! can not be known now. The constructors of these derived parameter
		//! classes can define the length of the vector.

		vector<double> _vector_of_parameters;

	}; // end of AbstractSquashingParameter

	ostream& operator<<(ostream&, const AbstractSquashingParameter&);
	istream& operator>>(istream&,       AbstractSquashingParameter&);

} // end of ImplementationLib

#endif // include guard
