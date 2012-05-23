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

#ifndef LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODECODE_H
#define LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODECODE_H

#include "FeatureMapNode.h"

using namespace LayerMappingLib;

template<class T, class F>
FeatureMapNode<T, F>::FeatureMapNode( vector<FeatureMapNode<T, F>* >& p, FeatureMap<T> a, F* function, const string& description, int x_skip, int y_skip, int input_rf_width, int input_rf_height, int output_rf_width, int output_rf_height, int input_padding_width, int input_padding_height ) :
	_activation( a ),
	_v_p_predecessor( p ),
	_function( function ),
	_description( description ),
	_x_skip( x_skip ),
	_y_skip( y_skip ),
	_input_rf_width( input_rf_width ),
	_input_rf_height( input_rf_height ),
	_output_rf_width( output_rf_width ),
	_output_rf_height( output_rf_height ),
	_input_padding_width( input_padding_width ),
	_input_padding_height( input_padding_height )
{
}

template<class T, class F>
FeatureMapNode<T, F>::FeatureMapNode() :
	_x_skip( -1 ),
	_y_skip( -1 ),
	_input_rf_width( -1 ),
	_input_rf_height( -1 ),
	_output_rf_width( -1),
	_output_rf_height( -1 ),
	_input_padding_width( -1 ),
	_input_padding_height( -1 )
{
}

template<class T, class F>
FeatureMapNode<T, F>::FeatureMapNode( const FeatureMapNode<T, F>& n ) :
	_activation( n._activation ),
	_v_p_predecessor( n._v_p_predecessor ),
	_function( n._function ),
	_description( n._description ),
	_x_skip( n._x_skip ),
	_y_skip( n._y_skip ),
	_input_rf_width( n._input_rf_width ),
	_input_rf_height( n._input_rf_height ),
	_output_rf_width( n._output_rf_width ),
	_output_rf_height( n._output_rf_height ),
	_input_padding_width( n._input_padding_width ),
	_input_padding_height( n._input_padding_height )
{
}

template<class T, class F>
FeatureMapNode<T, F>& FeatureMapNode<T, F>::operator=(  const FeatureMapNode<T, F>& n )
{
	_activation = n._activation;
	_v_p_predecessor = n._v_p_predecessor;
	_function = n._function;
	_description = n._description;
	_x_skip = n._x_skip;
	_y_skip = n._y_skip;
	_input_rf_width = n._input_rf_width;
	_input_rf_height = n._input_rf_height;
	_output_rf_width = n._output_rf_width;
	_output_rf_height = n._output_rf_height;
	_input_padding_width = n._input_padding_width;
	_input_padding_height = n._input_padding_height;
}

template<class T, class F>
void FeatureMapNode<T, F>::add_predecessors( vector<FeatureMapNode<T, F>* >& p )
{
	_v_p_predecessor.insert( _v_p_predecessor.end(), p.begin(), p.end() );
}

template<class T, class F>
void FeatureMapNode<T, F>::set_skip_size( int width, int height )
{
	_x_skip = width;
	_y_skip = height;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_receptive_field_size( int width, int height )
{
	_input_rf_width = width;
	_input_rf_height = height;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_output_receptive_field_size( int width, int height )
{
	_output_rf_width = width;
	_output_rf_height = height;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_input_padding_size( int width, int height )
{
	_input_padding_width = width;
	_input_padding_height = height;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_function( F* function )
{
	_function = function;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_description( const string& description )
{
	_description = description;
}

template<class T, class F>
int FeatureMapNode<T, F>::recpetive_field_witdh() const
{
	return _input_rf_width;
}

template<class T, class F>
int FeatureMapNode<T, F>::receptive_field_height() const
{
	return _input_rf_height;
}

template<class T, class F>
int FeatureMapNode<T, F>::x_skip() const
{
	return _x_skip;
}

template<class T, class F>
int FeatureMapNode<T, F>::y_skip() const
{
	return _y_skip;
}

template<class T, class F>
int FeatureMapNode<T, F>::output_rf_width() const
{
	return _output_rf_width;
}

template<class T, class F>
int FeatureMapNode<T, F>::output_rf_height() const
{
	return _output_rf_height;
}

template<class T, class F>
int FeatureMapNode<T, F>::input_padding_widht() const
{
	return _input_padding_width;
}

template<class T, class F>
int FeatureMapNode<T, F>::input_padding_height() const
{
	return _input_padding_height;
}

template<class T, class F>
int FeatureMapNode<T, F>::width() const
{
	return _activation.width();
}

template<class T, class F>
int FeatureMapNode<T, F>::height () const
{
	return _activation.height();
}

template<class T, class F>
F* FeatureMapNode<T, F>::function() const
{
	return _function;
}

template<class T, class F>
void FeatureMapNode<T, F>::update()
{
	if( this->has_predecessors() )
	{
		vector_list v_rf( nr_predecessors() );
		vector<iterator> pred_iter = predecessor_begin();

		//iterate through all nodes in output
		for( iterator a = this->activation_begin();
			a != this->activation_end();
			a++ )
		{
			//configure output receptive field.
			receptive_field output( &*a, 
					_output_rf_width,
					_output_rf_height,
					_activation.rowstride() );

			//configure input receptive fields
			typename vector_list::iterator i_v_rf = v_rf.begin();
			for( typename vector<iterator>::iterator pred = pred_iter.begin();
				pred != pred_iter.end();
				pred++, i_v_rf++ )
			{
				*i_v_rf = receptive_field( &**pred,
					_input_rf_width,
					_input_rf_height,
					_v_p_predecessor.front()->_activation.rowstride() );
				(*pred)++;
			}
			//apply function
			(*_function)( v_rf.begin(), v_rf.end(),
				output.begin(), output.end() );
		}
		#ifdef DEBUG
		//Assert that all predecessor iterators point to the end iterator.
		//If this assertion fails, the dimension of the output activation and number of receptive fields mismatch. If there is a receptive field for every ouput activation this assertion will hold.
		//If there is no mismatch this could point to an error in the implementation of iterator.
		vector<iterator> pred_end = predecessor_end();
		typename vector<iterator>::iterator b = pred_iter.begin();
		for( typename vector<iterator>::iterator e = pred_end.begin();
			e != pred_end.end();
			e++, b++ )
		{
			assert( *b == *e );
		}
		assert( b == pred_iter.end() );
		#endif //DEBUG
	}
}

template<class T, class F>
inline typename FeatureMapNode<T, F>::iterator FeatureMapNode<T, F>::activation_begin()
{
	return _activation.begin( _output_rf_width, _output_rf_height );
}

template<class T, class F>
inline typename FeatureMapNode<T, F>::iterator FeatureMapNode<T, F>::activation_end()
{
	return _activation.end( _output_rf_width, _output_rf_height );
}

template<class T, class F>
inline vector<typename FeatureMapNode<T, F>::iterator> FeatureMapNode<T, F>::predecessor_begin()
{
	vector<iterator> r;
	for( typename vector<FeatureMapNode<T, F>* >::iterator i = _v_p_predecessor.begin();
		i != _v_p_predecessor.end();
		i++ )
	{
		r.push_back(
			(*i)->activation().begin( _x_skip,
				_y_skip,
				_input_padding_width,
				_input_padding_height )
			);
	}
	return r;
}

template<class T, class F>
inline vector<typename FeatureMapNode<T, F>::iterator> FeatureMapNode<T, F>::predecessor_end()
{
	vector<iterator> r;
	for( typename vector<FeatureMapNode<T, F>* >::iterator i = _v_p_predecessor.begin();
		i != _v_p_predecessor.end();
		i++ )
	{
		r.push_back(
			(*i)->activation().end( _x_skip,
				_y_skip,
				_input_padding_width,
				_input_padding_height )
			);
	}
	return r;
}

template<class T, class F>
inline bool FeatureMapNode<T, F>::has_predecessors() const
{
	return _v_p_predecessor.size() > 0;
}

template<class T, class F>
inline int FeatureMapNode<T, F>::nr_predecessors() const
{
	return _v_p_predecessor.size();
}

template<class T, class F>
inline FeatureMap<T> FeatureMapNode<T, F>::activation()
{
	return _activation;
}

template<class T, class F>
void FeatureMapNode<T, F>::set_activation( FeatureMap<T> sl )
{
	_activation = sl;
}

template<class T, class F>
const string FeatureMapNode<T, F>::description() const
{
	return _description;
}

#ifdef DEBUG
template<class T, class F>
void FeatureMapNode<T, F>::debug_print()
{
	cout << "<FeatureMapNode>" << endl;
	if( _function != NULL )
	{
		_function->debug_print();
	}
	_activation.debug_print();
	cout << _description << endl;
	cout << "</FeatureMapNode>" << endl;
}
#endif //DEBUG

#endif //LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODECODE_H
