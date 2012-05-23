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
#ifndef _CODE_LIBS_POPULISTLIB_MUSIGMASCALARPRODUCT_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_MUSIGMASCALARPRODUCT_INCLUDE_GUARD

#include <vector>
#include <utility>
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "MuSigma.h"

using std::pair;
using std::vector;
using SparseImplementationLib::AbstractSparseNode;

namespace  PopulistLib {

	// forward declaration
	struct OrnsteinUhlenbeckConnection;

	//! Evaluates the scalar product of an input which arrives over OU_Connections.
	//! The formulae are:
	//! \f[
	//! \mu = N \tau \sum_i \nu_i J_i
	//! \f]
	//! and
	//! \f[
	//! \sigma^2 = N \tau \sum_i \nu_i J^2_i
	//! \f]
	class MuSigmaScalarProduct 
	{
	public:

		typedef pair<AbstractSparseNode<double,OrnsteinUhlenbeckConnection>*,OrnsteinUhlenbeckConnection> connection;
		typedef AbstractSparseNode<double,OrnsteinUhlenbeckConnection>::predecessor_iterator predecessor_iterator;

		//! Evaluate the inner product over connections which are indicated by the iterators
		MuSigma Evaluate
		(
			predecessor_iterator,	//!< begin iterator of the connection list
			predecessor_iterator,	//!< end iterator of the connection list
			Time					//!< membrane time constant
		) const;

	private:

		Potential InnerProduct
		(
			predecessor_iterator,
			predecessor_iterator
		) const ;

		Potential InnerSquaredProduct
		(
			predecessor_iterator,
			predecessor_iterator
		) const;

	};

} // Populist Lib


#endif // include guard
