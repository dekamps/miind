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

#ifndef LAYERMAPPINGNLIB_ABSTRACTFUNCTION_H
#define LAYERMAPPINGNLIB_ABSTRACTFUNCTION_H

#include <string>

using namespace std;

namespace LayerMappingLib
{
	/*! \class AbstractFunction
		\brief  AbstractFunction defines the interface for functions applied to receptive fields.

		The interface is pretty simple, the function operator is overloaded. A vector of iterators that point to	begin and end positions of the input receptive fields, and iterators that point to the begin and end position of the output receptive field are passed to the function.*/
	template<class VectorList>	
	struct AbstractFunction
	{
		/*! \brief A list of vectors 

			\f$(X^{d_1}\times X^{d_2} \times \ldots \times X^{d_n}) \f$ is called a list of vectors.*/
		typedef VectorList vector_list;
		/*! \brief An iterator, iterating through a list of vectors and pointing to a vector. */
		typedef typename VectorList::const_iterator vector_iterator;
		/*! \brief A Vector

			\f$ X^d \f$ is called a vector of dimension \f$ d \f$.*/
		typedef typename VectorList::value_type vector;
		/*! \brief An iterator, iterating through vector and pointing to elements of value_type. */
		typedef typename vector::iterator iterator;
		/*! \brief The basic data type.

			The set \f$ X \f$ is called value_type.*/
		typedef typename vector::value_type value_type;

		virtual ~AbstractFunction() {};

		/*! \brief The function operator.
			
			The function signature is given by \f$(X^{d_1}\times X^{d_2} \times \ldots \times X^{d_n}) \mapsto X^m \f$, where \f$ X \f$ is defined by value_type. Since the vector of input vectors is passed by iterators pointing to its begin and end position, \f$ n \f$ equals \code input_end - input_begin \endcode. The lenghts \f$ d_i \f$ of the input vectors can be mutually different and are defined by \code input[ i ].end() - input[ i ].begin() \endcode. \f$ m \f$ is defined by \code output_end - output_begin \endcode.

			\param input_begin A vector of iterators pointing to the begin positions of the input.
			\param input_end  A vector of iterators pointing to the end positions of the input.
			\param output_end An iterator pointing to the begin position of output.
			\param output_end An iterator pointing to the end position of output.*/
		virtual void operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end ) = 0;

		virtual int width() const;
		virtual int height() const;

		#ifdef DEBUG
		virtual void debug_print() const = 0;
		#endif //DEBUG
	};
}

template<class VectorList>
int AbstractFunction<VectorList>::width() const
{
	return 0;
}

template<class VectorList>
int AbstractFunction<VectorList>::height() const
{
	return 0;
}

#endif //LAYERMAPPINGNLIB_ABSTRACTFUNCTION_H
