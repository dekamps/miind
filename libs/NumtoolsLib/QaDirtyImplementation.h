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
#ifndef _CODE_LIBS_NUMTOOLSLIB_QADIRTYIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_QADIRTYIMPLEMENTATION_INCLUDE_GUARD

// Module: CODE.LIBS.NUMLIB
// Name:   QaDirty
// Author: Marc de Kamps (kamps@fsw.leidenuniv.nl)
// Created: 04-09-2002
//
// Short description:
// Implementation of QaDirty member functions
//----------------------------------------------------------------

#include <cassert>
#include <numeric>
#include "LocalDefinitions.h"
#include "NumtoolsLibException.h"
#include "QaDirty.h"


	namespace NumtoolsLib {

		//////////////////////////////////////////////////////////////////////////////////////////
		// qadirty inline function bodies
		//////////////////////////////////////////////////////////////////////////////////////////

		template <class V>
		inline QaDirty<V>::QaDirty
		( 
			long n_x, 
			long n_y 
		):
		_n_x_dim(n_x), 
		_n_y_dim(n_y), 
		_vec_data(n_x*n_y)
		{
		}

		template <class V>
		inline QaDirty<V>::QaDirty( const QaDirty<V>& rhs):
		_n_x_dim(rhs._n_x_dim),
		_n_y_dim(rhs._n_y_dim),
		_vec_data(rhs._vec_data) 
		{
		}

		template <class V>
		inline QaDirty<V>& QaDirty<V>::operator=(const QaDirty<V>& rhs)
		{
			if (this == &rhs)
				return *this;
			else 
			{
				this->_n_x_dim  = rhs._n_x_dim;
				this->_n_y_dim  = rhs._n_y_dim;
				this->_vec_data = rhs._vec_data;

				return *this;
			}

		}

		template <class Value>
		inline QaDirty<Value>::~QaDirty()
		{
		}

		template <class Value>
		inline Value& QaDirty<Value>::operator()
		(
			long i, 
			long j 
		)
		{
		        return _vec_data[i+_n_x_dim*j];
		}


		template <class Value>
		inline const Value& QaDirty<Value>::operator()
		(
			long i, 
			long j 
		) const
		{
				assert ( i < _n_x_dim );
				assert ( j < _n_y_dim );

		        return _vec_data[i+_n_x_dim*j];
		}

		template <class Value>
		inline long QaDirty<Value>::NrXdim() const
		{
		        return _n_x_dim;
		}

		template <class Value>
		inline long QaDirty<Value>::NrYdim() const
		{
		        return _n_y_dim;
		}

		template <class Value>
		inline void QaDirty<Value>::SetZero()
		{
			int n_elem = _n_x_dim*_n_y_dim;
			for (int i = 0; i < n_elem; i++)
				_vec_data[i] = 0;
		}

	} // end of NumtoolsLib.

#endif // include guard
