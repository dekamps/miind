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

#ifndef _CODE_LIBS_DYNAMICLIB_WILSONCOWANPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_WILSONCOWANPARAMETER_INCLUDE_GUARD

#include <iostream>
#include <boost/lexical_cast.hpp>
#include "../../UtilLib/UtilLib.h"
#include <MPILib/include/Basictypes.hpp>

using std::istream;
using std::ostream;


namespace MPILib
{
	//! WilsonCowanParameter
	struct WilsonCowanParameter
	{

		WilsonCowanParameter():
		_time_membrane(0),
		_rate_maximum(0),
		_f_noise(0),
		_f_input(0)
		{
		}

		//! constructor for convenience
		WilsonCowanParameter
		(
			//! membrane time constant in ms
			Time   time_membrane,

			//! maximum firing rate in Hz
			Rate   rate_maximum,

			//! noise parameter for sigmoid
			double f_noise,
		
			//! input
			double f_input = 0
				
		):	
		_time_membrane(time_membrane),
		_rate_maximum(rate_maximum),
		_f_noise(f_noise),
		_f_input(f_input)
		{
		}

		//! virtual destructor necessary: derives from Streamable
		virtual ~WilsonCowanParameter(){}


		//! membrane time constant
		Time _time_membrane;

		//! maximum firing rate
		Rate _rate_maximum;

		//! noise parameter
		double _f_noise;

		//! input
		double _f_input;
	};

} // end of DynamicLib

#endif // include guard