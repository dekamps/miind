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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_CONNECTIONISMLIB_TRAININGALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISMLIB_TRAININGALGORITHM_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <utility>

#ifdef WIN32
#pragma warning( disable: 4786 ) // uninteresting warning about the length of a name after
#endif							 // preprocessing

#include <memory>
#include "TrainingParameter.h"
#include "TrainingUnitCode.h"


using std::auto_ptr;
using NetLib::AbstractSquashingFunction;

namespace ConnectionismLib
{

	//! Abstract base class for TrainingAlgorithms

	template <class Implementation>
	class TrainingAlgorithm {
	public:

		typedef Implementation*						implementation_pointer;
		typedef TrainingAlgorithm<Implementation>	training;

		virtual			~TrainingAlgorithm() = 0;

		//! Training takes in put pattern, or an output pattern, or both
		virtual bool	Train(const TrainingUnit<typename Implementation::NodeValue>&) = 0;

		//! Almost all TrainingAlgorithms require an explicit initialization step
		virtual bool    Initialize() = 0;

		//!  A user configures a training network, the network uses Clone to generates its own TrainingObject
		//! and to pass it implementation details, like implementation_pointer etc.
		virtual TrainingAlgorithm*
						Clone(implementation_pointer) const = 0;
	private:

	}; // end of TrainingAlgorithm

	template <class Implementation>
	inline TrainingAlgorithm<Implementation>::~TrainingAlgorithm()
	{
	}


	class NoTrainingIntended {
	};

} // end of Connectionism

#endif //include guard 
