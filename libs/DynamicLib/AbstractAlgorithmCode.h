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

#ifndef _CODE_LIBS_DYNAMICLIB_ABSTRACTALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_ABSTRACTALGORITHMCODE_INCLUDE_GUARD

#include "AbstractAlgorithm.h"
#include "DynamicLibException.h"

namespace DynamicLib
{
	template <class WeightValue>
	AbstractAlgorithm<WeightValue>::AbstractAlgorithm
	(
		Number number_of_elements
	)
	{
	}

	template <class WeightValue>
	AbstractAlgorithm<WeightValue>::AbstractAlgorithm
	(
		const AbstractAlgorithm<WeightValue>& rhs
	):
	Persistant(rhs)
	{	
	}

	template <class WeightValue>
	AbstractAlgorithm<WeightValue>::~AbstractAlgorithm()
	{
	}

	template <class WeightValue>
	string AbstractAlgorithm<WeightValue>::Tag() const
	{
		return STR_AE_TAG;
	}

	template <class WeightValue>
	bool AbstractAlgorithm<WeightValue>::FromStream(istream& s)
	{
		throw DynamicLibException(STR_AE_EXCEPTION);
		return true;
	}

	template <class WeightValue>
	bool AbstractAlgorithm<WeightValue>::ToStream(ostream& s) const
	{
		s << Tag()      << "\n";
		s << ToEndTag(Tag()) << "\n";
		return true;
	}

	template <class WeightValue>
	double AbstractAlgorithm<WeightValue>::InnerProduct
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	) const
	{
		// No input nodes, no input. MSVC10 is fuzzy about p_begin and p_end de facto being 0 pointers
		if (iter_begin == iter_end )
			return 0;

		Connection* p_begin = iter_begin.ConnectionPointer();
		Connection* p_end   = iter_end.ConnectionPointer();

		return inner_product
			(
				p_begin, 
				p_end,   
				p_begin, 
				0.0,
				plus<double>(),
				SparseImplementationLib::ConnectionProduct<double,WeightValue>()
			);
	}

	template <class WeightValue>
	valarray<double>& AbstractAlgorithm<WeightValue>::ArrayState(AlgorithmGrid& grid) const
	{
		return grid.ArrayState();
	}

	template <class WeightValue>
	valarray<double>& AbstractAlgorithm<WeightValue>::ArrayInterpretation(AlgorithmGrid& grid) const
	{
		return grid.ArrayInterpretation();
	}

	template <class WeightValue>
	Number& AbstractAlgorithm<WeightValue>::StateSize(AlgorithmGrid& grid) const 
	{
		return grid.StateSize();
	}

	template <class WeightValue>
	Number AbstractAlgorithm<WeightValue>::StateSize(const DynamicLib::AlgorithmGrid & grid) const
	{
		return grid.StateSize();
	}

} // end of DynamicLib

#endif // include guard
