// Copyright (c) 2005 - 2007 Marc de Kamps, Johannes Drever, Melanie Dietz
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

#ifndef LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXREAL_H
#define LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXREAL_H

#include "../AbstractFunction.h"

#include <assert.h>

namespace LayerMappingLib
{
	template<class VectorList>
	struct ProductComplexReal : public AbstractFunction<VectorList>
	{
		typedef typename AbstractFunction<VectorList>::vector_list vector_list;
		typedef typename AbstractFunction<VectorList>::vector_iterator vector_iterator;
		typedef typename AbstractFunction<VectorList>::vector vector;
		typedef typename AbstractFunction<VectorList>::iterator iterator;
		typedef typename AbstractFunction<VectorList>::value_type value_type;

		virtual ~ProductComplexReal() {};

		virtual void operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end );

		#ifdef DEBUG
		virtual void debug_print() const;
		#endif //DEBUG

		private:
		string _description;
	};
}

#endif //LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXREAL_H
