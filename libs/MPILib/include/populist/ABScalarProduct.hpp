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
#ifndef MPILIB_POPULIST_ABSCALARPRODUCT_HPP_
#define MPILIB_POPULIST_ABSCALARPRODUCT_HPP_

#include <MPILib/include/populist/ABStruct.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/utilities/Exception.hpp>

namespace MPILib {
namespace populist {

	class ABScalarProduct 
	{
	public:

		typedef pair<AbstractSparseNode<double,OrnsteinUhlenbeckConnection>*,OrnsteinUhlenbeckConnection> connection;
		typedef AbstractSparseNode<double,OrnsteinUhlenbeckConnection>::predecessor_iterator predecessor_iterator;

		//! Evaluate the inner product over connections which are indicated by the iterators
		ABQStruct Evaluate
		(
			predecessor_iterator iter_begin,	//!< begin iterator of the connection list
			predecessor_iterator iter_end,		//!< end iterator of the connection list
			Time								//!< membrane time constant
		) const
		{
			// for now a comes from the first population and b from the second. This will need to change.
			ABQStruct par_ret;

			if ( iter_begin->GetValue() == 6.0 )
			{
				par_ret._a = 6.91423056;
				par_ret._b = 0.13299526;
			}
			else
				if (iter_begin->GetValue() == 8.0)
				{
					par_ret._a = 129.43365395;
					par_ret._b = 0.08430153;
				}
				else
					throw utilities::Exception("Input rate cannot be decoded by ABScalarProduct");

			return par_ret;
		}
	private:

	};

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_ABSCALARPRODUCT_HPP_
