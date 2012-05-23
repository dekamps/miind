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
#ifndef _CODE_LIBS_DYNAMCLIB_CANVASPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMCLIB_CANVASPARAMETER_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"

using UtilLib::Streamable;

namespace DynamicLib {

	//! Auxiliarry class, stores the boundaries of the histograms shown in the running canvas. 

	//! This class was created so that the canvas can be configured from an XML file. It controls the
	//! minimum state, the maximum state, the minimum and maximum frequency as they are shown in the
	//! running ROOt canvas, but does not influence the simulation in any way.
	struct CanvasParameter : public Streamable{

		//! default constructor
		CanvasParameter();

		CanvasParameter
		(
			double,		//!< t_min
			double,		//!< t_max
			double,		//!< f_min
			double,     //!< f_max
			double,		//!< state_min
			double,		//!< state_max
			double,		//!< dense_min, the minimum value shown in the state diagram
			double		//!< dense_max, the maximum value shown in the state diagram
		);

		//! virtual destructor
		virtual ~CanvasParameter();

		//! tag for serialization
		virtual string Tag() const;

		//! streaming output
		virtual bool ToStream(ostream&) const;

		//! streaming input
		virtual bool FromStream(istream&);

		double _t_min;
		double _t_max;
		double _f_min;
		double _f_max;
		double _state_min;
		double _state_max;
		double _dense_min;
		double _dense_max;
	};

	const CanvasParameter 
		DEFAULT_CANVAS 
		(
			0.,		//!< t_min
			1.,		//!< t_max
			0.,		//!< f_min
			20.,	//!< f_max
			0.,		//!< v_min
			1.,		//!< v_max
			0.,		//!< rho_min
			3.0		//!< rho_max
		);

}

#endif // include guard
