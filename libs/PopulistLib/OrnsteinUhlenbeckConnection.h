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
#ifndef _CODE_LIBS_POPULISTLIB_ORNSTEINUHLENBECKCONNECTION_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_ORNSTEINUHLENBECKCONNECTION_INCLUDE_GUARD

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"
#include "MuSigmaScalarProduct.h"
using std::ostream;


namespace PopulistLib {

	//! Connection parameters for a network of OrnsteinUhlenbeck populations
	//! need two parameters: a number parameter, which is not squared in the
	//! calculation of sigma, and an efficacy, which is squared.
	struct OrnsteinUhlenbeckConnection {

		//! In PopulistLib the connections have an associated inner product
		typedef MuSigmaScalarProduct InnerProduct;

		//! effective number, may be fractional
		double		_number_of_connections;

		//! effective synaptic efficacy from one population on another
		Efficacy	_efficacy;

		//! delay of a connection
		Time		_delay;

		//! default constructor
		OrnsteinUhlenbeckConnection
		(
		):
		_number_of_connections(0),
		_efficacy(0),
		_delay(0)
		{
		}

		//! construct, using effective number of connections and effectivie efficacy
		OrnsteinUhlenbeckConnection
		(
			double		number_of_connections,	//!< effective number of connections
			Efficacy	efficacy,				//!< synaptic efficacy
			Time		delay	= 0				//!< delay of connection
		):
		_number_of_connections(number_of_connections),
		_efficacy(efficacy),
		_delay(delay)
		{
		}
	};

	typedef OrnsteinUhlenbeckConnection PopulationConnection;

	//! Necessary for a sensible definition of InnerProduct
	inline double operator*
	(
		double f, 
		const OrnsteinUhlenbeckConnection& connection
	)
	{
		return f*connection._efficacy*connection._number_of_connections;
	}

	//! Necessary for serialization of Networks
	inline ostream& operator<<
	(
		ostream& s, 
		const OrnsteinUhlenbeckConnection& connection
	)
	{
		s << connection._number_of_connections << " " << connection._efficacy << " " << connection._delay;
		return s;
	}

	inline istream& operator>>
	(
			istream& s, 
			OrnsteinUhlenbeckConnection& connection
	)
	{
		s >> connection._number_of_connections >> connection._efficacy >> connection._delay;
		return s;
	}

	//! Necessary for tests on Dale's law
	inline double ToEfficacy
	( 
		const OrnsteinUhlenbeckConnection& connection
	)
	{ 
		return connection._efficacy; 
	}

	typedef OrnsteinUhlenbeckConnection OU_Connection;

}

inline int IsNan
(
	const PopulistLib::OrnsteinUhlenbeckConnection& connection
)
{
	return IsNan(connection._efficacy) + IsNan(connection._number_of_connections) +IsNan(connection._delay);
}



#endif // include guard
