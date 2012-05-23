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
#ifndef _CODE_LIBS_POPULISTLIB_CONNECTIONSQUAREDPRODUCT_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_CONNECTIONSQUAREDPRODUCT_INCLUDE_GUARD

#include <utility>
#include <functional>
#include <vector>
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "OrnsteinUhlenbeckConnection.h"

using std::binary_function;
using std::pair;
using std::vector;
using SparseImplementationLib::SparseNode;

namespace PopulistLib {


	class ConnectionSquaredProduct : public binary_function<OrnsteinUhlenbeckConnection, OrnsteinUhlenbeckConnection, OrnsteinUhlenbeckConnection>
	{
	public:

		typedef pair<AbstractSparseNode<double, OrnsteinUhlenbeckConnection>*, OrnsteinUhlenbeckConnection> connection;

		inline double operator()
		(
			connection connection_first,
			connection connection_second
		) const
		{
			double f_node_rate = connection_first.first->GetValue();
			double f_efficacy_squared = connection_second.second._efficacy*connection_second.second._efficacy;
			double f_number = connection_second.second._number_of_connections;

			return f_node_rate*f_efficacy_squared*f_number;
		}
	};

}

#endif // include guard
