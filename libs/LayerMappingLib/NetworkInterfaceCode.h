#ifndef LAYERMAPPINGLIB_NETWORKINTERFACECODE_H
#define LAYERMAPPINGLIB_NETWORKINTERFACECODE_H

#include "NetworkInterface.h"

using namespace LayerMappingLib;

template<class Network>
NetworkInterface<Network>::NetworkInterface( Network n ) : _network( n )
{
}

template<class Network>
Network& NetworkInterface<Network>::network()
{
	return _network;
}

template<class Network>
void NetworkInterface<Network>::evolve()
{
	LayerMappingLib::evolve( _network.begin(), _network.end() );
}

template<class Network>
void NetworkInterface<Network>::evolve( int i )
{
	LayerMappingLib::evolve( _network.begin( i ), _network.end( i ) );
}

template<class Network>
void NetworkInterface<Network>::set_input( const std::vector<double>& image )
{
	//set input activation to image
	//TODO: originally an array with variable size was declared here. Amazingly that compiled until recently. It has now
	// been replaced with the following construction, but this has not been tested yet. (MdK: 25/06/11)
	boost::shared_ptr<double> t = boost::shared_ptr<double>(new double[image.size()]);
	
	copy( image.begin(), image.end(), t.get() );

	_network.begin()->layer_activation( 0 ).front().get( t.get() );
}

template<class Network>
void NetworkInterface<Network>::set_input( double* image )
{
	_network.begin()->layer_activation( 0 ).front().get( image );
}

template<class Network>
void NetworkInterface<Network>::set_feedback_template( double* feedback_template )
{
	NetworkEnsemble<double>::network feedback = *( ++_network.begin() );
	cout << "This function is for debug porposes only. Application may crash if used with a wrong network ensemble." << endl;

	int layer = 0;
	double* fb = feedback_template;
	for( NetworkEnsemble<double>::network::iterator i = feedback.begin( layer );
		i != feedback.end( layer );
		i++, fb++ )
	{
		double t[ 1 ];
		*t = *fb;
// 		fill( t, t + i->width() * i->height(), 1 );
		i->activation().get( t );
	}
// 	assert( fb == feedback_template.end() );
}

template<class Network>
void NetworkInterface<Network>::set_feedback_template( const std::vector<double>& feedback_template )
{
	NetworkEnsemble<double>::network feedback = *( ++_network.begin() );
	cout << "This function is for debug porposes only. Application may crash if used with a wrong network ensemble." << endl;

	int layer = 0;
	vector<double>::const_iterator fb = feedback_template.begin();
	for( NetworkEnsemble<double>::network::iterator i = feedback.begin( layer );
		i != feedback.end( layer );
		i++, fb++ )
	{
		double t[ 1 ];
		*t = *fb;
// 		fill( t, t + i->width() * i->height(), 1 );
		i->activation().get( t );
	}
	assert( fb == feedback_template.end() );
}

template<class Network>
vector<double> NetworkInterface<Network>::feature_vector()
{
	vector<value_type> r;

	int lastlayer = _network.begin()->nr_layers() - 1;
	for( typename Network::network::iterator i = _network.begin()->begin( lastlayer );
		i != _network.begin()->end( lastlayer );
		i++ )
	{
		r.push_back( *(i)->activation().begin() );
	}

	return r;
}

template<class Network>
vector<double> NetworkInterface<Network>::feature_vector( int n )
{
	vector<value_type> r;

	int lastlayer = n;
	for( typename Network::network::iterator i = _network.begin()->begin( lastlayer );
		i != _network.begin()->end( lastlayer );
		i++ )
	{
		typename Network::network::node::iterator b = i->activation().begin();
		typename Network::network::node::iterator e = i->activation().end();

// 		cout <<  std::accumulate( b, e, 0.0 ) << " " << i->width() << " " << i->height() <<endl;
		r.push_back( std::accumulate( b, e, 0.0 ) / ( i->width() * i->height() ) );
	}

	return r;
}

template<class Network>
int NetworkInterface<Network>::nr_feature_maps( int network, int layer )
{
	typename Network::network ne = *( _network.begin() + network );
	return ne.nr_feature_maps( layer );
}

template<class Network>
vector<double> NetworkInterface<Network>::feature_map( int n, int l, int fm )
{
	typedef typename Network::network::node::iterator node_iterator;

	assert( n < ( _network.end() - _network.begin() ) );
	typename Network::network ne = *( _network.begin() + n );

	assert( l < ne.nr_layers() );
	assert( fm < ne.nr_feature_maps( l ) );
// 	assert( fm < ( ne.end( l ) - ne.begin( l ) ) );
	typename Network::network::node node = * ( ne.begin( l ) + fm );

	int width = node.activation().width();
	int height = node.activation().height();

	vector<double> r( width * height );

	vector<double>::iterator x = r.begin();
	
	for( node_iterator i = node.activation().begin();
		i != node.activation().end();
		i++, x++ )
	{
		*x = *i;
	}
	return r;
}

template<class Network>
void NetworkInterface<Network>::fill_feature_map_padding_with_noise( double level )
{
	for( typename Network::iterator i = _network.begin();
		i != _network.end();
		i++ )
	{
		i->fill_feature_map_padding_with_noise( level );
	}
}

template<class Network>
pair<double, double> NetworkInterface<Network>::feature_map_size( int n, int l, int fm )
{
	typedef typename Network::network::node::iterator node_iterator;

	assert( n < ( _network.end() - _network.begin() ) );
	typename Network::network ne = *( _network.begin() + n );

	assert( l < ne.nr_layers() );
	assert( fm < ne.nr_feature_maps( l ) );
// 	assert( fm < ( ne.end( l ) - ne.begin( l ) ) );
	typename Network::network::node node = *( ne.begin( l ) + fm );


	pair<double, double> size;

	size.first = node.activation().width();
	size.second = node.activation().height();

	return size;
}


#endif //LAYERMAPPINGLIB_NETWORKINTERFACECODE_H
