// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_NUMTOOLSLIB_QADIRTY_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_QADIRTY_INCLUDE_GUARD

// Module: CODE.LIBS.NUMLIB
// Name:   QaDirty
// Author: Marc de Kamps (kamps@fsw.leidenuniv.nl)
// Created: 3-09-2002
//
// Short description:
// Straightforward implementation of a matrix class
//----------------------------------------------------------------

#include <string>
#include <vector>
#include <complex>

using std::complex;
using std::string;
using std::vector;

namespace NumtoolsLib 
{

	template <class Value>
	class QaDirty {

	public:
	
		typedef       Value&          reference;
		typedef const Value&          const_reference;
		typedef       QaDirty<Value>& matrix_reference;
		typedef const QaDirty<Value>& const_matrix_reference;

	// ctor's and the like:

	 QaDirty(long, long);                               // standard ctor

	 QaDirty(const QaDirty<Value>&);                        // copy ctor
	~QaDirty();                                             // dtor
		 
	matrix_reference operator=( const_matrix_reference );  // asignment operator

	void SetZero();

	// element acces:

	reference       operator()(long, long);
	const_reference operator()(long, long) const;

	    // dimensions:

	long NrXdim() const;
	long NrYdim() const;

		 string Name() const { return string("Quick and Dirty"); }

		// numerical operators


	private:

		long 			_n_x_dim;		// number of colums
		long 			_n_y_dim;		// number of rows
		vector<Value>	_vec_data;			// pointer to data storage

	}; // end of QaDirty.


	typedef QaDirty<double>           D_Matrix;
	typedef QaDirty<float>            F_Matrix;
	typedef QaDirty<complex<double> > C_Matrix;


} // end of Numtools.

	


#endif // include guard

