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

#ifndef LAYERMAPPINGLIB_FEATUREMAPNETWORKCODE_H
#define LAYERMAPPINGLIB_FEATUREMAPNETWORKCODE_H

#include "FeatureMapNetwork.h"

using namespace LayerMappingLib;

template<class T>
FeatureMapNetwork<T>::FeatureMapNetwork()
{
}

template<class T>
FeatureMapNetwork<T>::FeatureMapNetwork( int nr_layer, const string& title ) : _v_layer_size( nr_layer, 0 ), _title( title )
{
	for( int i = 0; i < nr_layer; i++ )
	{
		stringstream ss( stringstream::in | stringstream::out );
		ss << "Layer " << i;
		_v_layer_name.push_back( ss.str() );
	}
}

template<class T>
FeatureMapNetwork<T>::FeatureMapNetwork( FeatureMapNetwork<T>& n1, FeatureMapNetwork<T>& n2, const string& title )
{
	assert( n1.nr_layers() == n2.nr_layers() );

	int original_nr_layer = n1.nr_layers();

	FeatureMapNetwork<T> network( original_nr_layer * 2, title );

	for( int i = 0;
		i != original_nr_layer;
		i++ )
	{
		int factor_layer = i * 2;
		int product_layer = ( i * 2 ) + 1;

		stringstream ss( stringstream::in | stringstream::out );
		ss << "Factor layer" << i;
		network.set_layer_name( factor_layer, ss.str() );
		ss.str( "" );
		ss << "Product layer" << i;
		network.set_layer_name( product_layer, ss.str() );

		iterator f1 = n1.begin( i );
		for( iterator f2 = n2.begin( original_nr_layer - 1 - i );
			f2 != n2.end( original_nr_layer - 1 - i );
			f1++, f2++ )
		{
			int width =  f1->activation().width();
			int height = f1->activation().height();
			assert( f2->activation().width() == width );
			assert( f2->activation().height() == height );

			vector<node*> p;
			p.push_back( network.add_node( f1->activation(),
				factor_layer,
				f1->description() ) );
			p.push_back( network.add_node( f2->activation(),
				factor_layer,
				f2->description()
				) );

			stringstream s( stringstream::in | stringstream::out );
			s << PRODUCT;
			ss.str( "" );
			ss << f1->description() << "*" << f2->description() << "-" << width << "x" << height;
			network.add_node( p, 
				FeatureMap<T>( width, height, 0, 0 ),
				function_factory::get_function( s ),
				product_layer,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				ss.str() );
		}
		assert( f1 == n1.end( i ) );
	}

	*this = network;
}

template<class T>
FeatureMapNetwork<T>::~FeatureMapNetwork()
{
	for_each( _v_node.begin(),
		_v_node.end(),
		boost::lambda::bind( boost::lambda::delete_ptr(), boost::lambda::_1 ) );
}

template<class T>
FeatureMapNetwork<T>::FeatureMapNetwork( const FeatureMapNetwork<T>& n ) :
		_v_node( _clone_nodes( n ) ),
		_v_layer_size( n._v_layer_size ),
		_v_layer_name( n._v_layer_name ),
		_title( n._title )
{
}

template<class T>
FeatureMapNetwork<T> FeatureMapNetwork<T>::clone( const string& title )
{
	FeatureMapNetwork network( nr_layers(), title );

	for( int i = 0; i < nr_layers(); i++ )
	{
		for( iterator n = begin( i );
			n != end( i );
			n++ )
		{
			network.add_node( FeatureMap<T>( n->activation().width(), n->activation().height(), 0, 0 ),
				nr_layers() - 1 - i, n->description() );
		}
	}

	return network;
}

template<class T>
typename FeatureMapNetwork<T>::iterator FeatureMapNetwork<T>::node_at( int layer, int i )
{
	return this->begin( layer ) + i;
}

template<class T>
void FeatureMapNetwork<T>::connect( FeatureMapNetwork<T>& n )
{
	iterator ti = this->begin( this->nr_layers() - 1 );
	for( iterator i = n.begin( n.nr_layers() - 1 );
		i != n.end( n.nr_layers() - 1 );
		i++, ti++ )
	{
		ti->set_activation( i->activation() );
	}
	assert( ti == this->end( this->nr_layers() - 1 ) );
}

template<class T>
vector<typename FeatureMapNetwork<T>::node*> FeatureMapNetwork<T>::_clone_nodes( const FeatureMapNetwork<T>& n )
{
	map<node*, node*> m;

	vector<node*> v_node;

	for( typename vector<node*>::const_iterator ni = n._v_node.begin();
		ni != n._v_node.end();
		ni++ )
	{
		node* pn =  new node( *(*ni) );

		v_node.push_back( pn );
		
		m[ *ni ] = pn;
	}

	typename vector<node*>::const_iterator ni = n._v_node.begin();
	for( typename vector<node*>::iterator i = v_node.begin();
		i != v_node.end();
		i++, ni++ )
	{
		assert( *i = m[ *ni ] );
		typename vector<node*>::iterator np = (*ni)->_v_p_predecessor.begin();
		for( typename vector<node*>::iterator p = (*i)->_v_p_predecessor.begin();
			p != (*i)->_v_p_predecessor.end();
			p++, np++ )
		{
			*p = m[ *np ];
		}
		assert( np == (*ni)->_v_p_predecessor.end() );
	}
	assert( ni == n._v_node.end() );

	return v_node;
}

template<class T>
FeatureMapNetwork<T>& FeatureMapNetwork<T>::operator=( const FeatureMapNetwork<T>& n )
{
	_v_node = _clone_nodes( n );
	_v_layer_name = n._v_layer_name;
	_v_layer_size = n._v_layer_size;
	_title = n._title;

	return *this;
}

template<class T>
vector<typename FeatureMapNetwork<T>::node*> FeatureMapNetwork<T>::add_simple_cell_layer( int preceding_layer_number, int layer_number, const vector<function*>& functions, const vector<string>& descriptions, vector<int>& padding_width, vector<int>& padding_height )
{
	#ifdef DEBUG
	cout << "adding " << _v_layer_name.at( layer_number ) << endl;
	int j = 0;
	for( iterator i = this->begin( preceding_layer_number );
		i != this->end( preceding_layer_number );
		i++, j++ )
	{
		cout << "\t\t" << i->activation().width() << "x" << i->activation().height() << "\t" << padding_width.at( j ) << "x" << padding_height.at( j ) << "\t" << 1 << "x" << 1 << "\t1x1\t" << endl;
	}
	#endif

	vector<int>::iterator pwi = padding_width.begin();
	vector<int>::iterator phi = padding_height.begin();

	vector<node*> layer;
	for( int i = 0;
		i < this->nr_feature_maps( preceding_layer_number );
		i++ )
	{
		vector<string>::const_iterator description = descriptions.begin();
		for( typename vector<function*>::const_iterator fni = functions.begin();
			fni != functions.end();
			fni++, description++, pwi++, phi++ )
		{
			function* filter = *fni;
			int receptive_field_width = filter->width();
			int receptive_field_height = filter->height();

			vector<node*> predecessors;
			predecessors.push_back( &( *this->node_at( preceding_layer_number, i ) ) );
			layer.push_back( add_node( predecessors,
				FeatureMap<T>( this->node_at( preceding_layer_number, i 	)->activation().width(), this->node_at( preceding_layer_number, i )->activation().height(),
					*pwi, *phi ),
				filter,
				layer_number,
				1, 1,
				receptive_field_width, receptive_field_height,
/*				( receptive_field_width - ( receptive_field_width % 2 ) ) / 2, ( receptive_field_height - ( receptive_field_height % 2 )  ) / 2,*/
				receptive_field_width / 2, receptive_field_height / 2,	
				1, 1,
				*description ) );
		}
		assert( description == descriptions.end() );
	}
	assert( pwi == padding_width.end() );
	assert( phi == padding_height.end() );

	return layer;
}

template<class T>
vector<typename FeatureMapNetwork<T>::node*> FeatureMapNetwork<T>::add_complex_cell_layer(
	int preceding_layer_number, int layer_number,
	const vector<function*> pooling_functions, const vector<string>& descriptions,
	int offset,
	const vector<int>& filter_bands,
	const vector<int>& receptive_field_width, const vector<int>& receptive_field_height,
	const vector<int>& skip_width, const vector<int>& skip_height,
	const vector<int>& padding_width, const vector<int>& padding_height )
{
// 	cout << "nr_predecessors: 	" << this->nr_feature_maps( preceding_layer_number ) << endl;
// 
// 	cout << "Layers: 	" << preceding_layer_number << " " << layer_number << endl;
// 	copy( pooling_functions.begin(), pooling_functions.end(), ostream_iterator<function*>( cout, " " ) ); cout << endl;
// 	cout << "offset:	" << offset << endl;
// 	cout << "Filterbands:" << endl;
// 	copy( filter_bands.begin(), filter_bands.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "receptive field width:" << endl;
// 	copy( receptive_field_width.begin(), receptive_field_width.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "receptive field height:" << endl;
// 	copy( receptive_field_height.begin(), receptive_field_height.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "skip_width:" << endl;
// 	copy( skip_width.begin(), skip_width.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "skip_height:" << endl;
// 	copy( skip_height.begin(), skip_height.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "padding_width:" << endl;
// 	copy( padding_width.begin(), padding_width.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "padding_height:" << endl;
// 	copy( padding_height.begin(), padding_height.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;


	vector<node*> layer;
	int fb_offset = 0;

	vector<int>::const_iterator skip_width_iter = skip_width.begin();
	vector<int>::const_iterator skip_height_iter = skip_height.begin();
	vector<int>::const_iterator receptive_field_width_iter = receptive_field_width.begin();
	vector<int>::const_iterator receptive_field_height_iter = receptive_field_height.begin();
	vector<int>::const_iterator padding_width_iter = padding_width.begin();
	vector<int>::const_iterator padding_height_iter = padding_height.begin();
	typename vector<function*>::const_iterator pooling_function = pooling_functions.begin();
	vector<string>::const_iterator description = descriptions.begin();
	for( vector<int>::const_iterator filter_band = filter_bands.begin();
		filter_band != filter_bands.end();
		filter_band++, skip_width_iter++, skip_height_iter++, receptive_field_width_iter++, receptive_field_height_iter++, padding_width_iter++, padding_height_iter++, pooling_function++, description++ )
	{
		vector<node*> predecessors;	
		for( int i = 0;
			i < offset;
			i++ )
		{
			for( int filter = 0;
				filter < *filter_band;
				filter++ )
			{
				predecessors.push_back( &( *this->node_at(
					preceding_layer_number,
					i + ( filter + fb_offset ) * offset ) ) );
			}
			int width = (int) ceil( (double) predecessors.front()->activation().width() / (double) *skip_width_iter );
			int height = (int) ceil( (double) predecessors.front()->activation().height() / (double) *skip_height_iter );

			layer.push_back( this->add_node( predecessors,
				FeatureMap<T>( width, height, *padding_width_iter, *padding_height_iter ),
				*pooling_function,
				layer_number,
				*skip_width_iter, *skip_height_iter,
				*receptive_field_width_iter, *receptive_field_height_iter,
				*padding_width_iter, *padding_height_iter,
				1, 1,
				*description ) );
		}
		fb_offset += *filter_band;
	}
	assert( skip_width_iter == skip_width.end() );
	assert( skip_height_iter == skip_height.end() );
	assert( receptive_field_width_iter == receptive_field_width.end() );
	assert( receptive_field_height_iter == receptive_field_height.end() );
	assert( padding_width_iter == padding_width.end() );
	assert( padding_height_iter == padding_height.end() );
	assert( pooling_function == pooling_functions.end() );
	assert( description == descriptions.end() );

	return layer;
}

template<class T>
FeatureMapNode<T, typename FeatureMapNetwork<T>::function>* FeatureMapNetwork<T>::add_node( FeatureMap<T> activation, int layer , const string& node_description )
{
	vector<node*> null;

	node* n = new node( null,
			activation,
			function_factory::empty_function(),
			node_description,
			1, 1,
			0, 0,
			0, 0,
			0, 0 );

	_v_node.insert( _v_node.begin() + accumulate( _v_layer_size.begin(),
						_v_layer_size.begin() + layer + 1,
						0 ),
					n );
		
	_v_layer_size.at( layer )++;

	return n;
}

template<class T>
FeatureMapNode<T, typename FeatureMapNetwork<T>::function>* FeatureMapNetwork<T>::add_node( vector<FeatureMapNode<T, function>* >& predecessors, FeatureMap<T> activation, function* f, int layer_number, int x_skip, int y_skip, int input_rf_width, int input_rf_height, int input_padding_width, int input_padding_height, int output_rf_width, int output_rf_height, const string& node_description )
{
	_v_node.insert( _v_node.begin() + accumulate( _v_layer_size.begin(),
						_v_layer_size.begin() + layer_number + 1,
						0 ),
		new FeatureMapNode<T, function>( predecessors,
			activation,
			f,
			node_description,
			x_skip, y_skip,
			input_rf_width, input_rf_height,
			output_rf_width, output_rf_height,
			input_padding_width, input_padding_height ) );

	_v_layer_size.at( layer_number )++;

	return _v_node.back();
}

template<class T>
const string FeatureMapNetwork<T>::title() const
{
	return _title;
}

template<class T>
typename FeatureMapNetwork<T>::iterator FeatureMapNetwork<T>::begin()
{
	return iterator( _v_node.begin() );
}

template<class T>
typename FeatureMapNetwork<T>::iterator FeatureMapNetwork<T>::end()
{
	return iterator( _v_node.end() );
}

template<class T>
vector<FeatureMap<T> > FeatureMapNetwork<T>::layer_activation( int i)
{
	vector<FeatureMap<T> > v;

	for( iterator it = begin( i );
		it != end( i );
		it++ )
	{
		v.push_back( it->activation() );
	}
	return v;
}

template<class T>
void FeatureMapNetwork<T>::update()
{
// 	evolve( begin(), end() );
	for( iterator i = begin();
		i != end();
		i++ )
	{
// 		cout << i->description() << endl;
		i->update();
	}
// 	cout << endl;
}

template<class T>
typename FeatureMapNetwork<T>::iterator FeatureMapNetwork<T>::begin( int i )
{
	return iterator( _v_node.begin() + std::accumulate( _v_layer_size.begin(), _v_layer_size.begin() + i, 0 ) );
}

template<class T>
typename FeatureMapNetwork<T>::iterator FeatureMapNetwork<T>::end( int i )
{
	return iterator( _v_node.begin() + accumulate( _v_layer_size.begin(), _v_layer_size.begin() + ( i + 1 ), 0 ) );
}

template<class T>
const string& FeatureMapNetwork<T>::layer_name( int i ) const
{
	return _v_layer_name.at( i );
}

template<class T>
void FeatureMapNetwork<T>::set_layer_name( int i, const string& s )
{
	_v_layer_name.at( i ) = s;
}

template<class T>
int FeatureMapNetwork<T>::nr_layers() const
{
	return _v_layer_size.size();
}

template<class T>
int FeatureMapNetwork<T>::nr_feature_maps( int i ) const
{
	return _v_layer_size.at( i );
}

template<class T>
void FeatureMapNetwork<T>::fill_feature_map_padding_with_noise( double level )
{
	for( iterator i = begin();
		i != end();
		i++ )
	{
		i->activation().fill_padding_with_noise( level );
	}
}

#ifdef DEBUG
template<class T>
void FeatureMapNetwork<T>::debug_print()
{
	cout << "<FeatureMapNetwork>" << endl;
	for( typename vector<FeatureMapNode<T, function>* >::iterator node = _v_node.begin();
		node != _v_node.end();
		node++ )
	{
		(*node)->debug_print();
	}
	cout << "</FeatureMapNetwork>" << endl;
}
#endif //DEBUG

#endif //LAYERMAPPINGLIB_FEATUREMAPNETWORKCODE_H
