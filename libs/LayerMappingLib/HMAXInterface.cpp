#include "HMAXInterface.h"

using namespace LayerMappingLib;

HMAXInterface::HMAXInterface( Network n ) : NetworkInterface<NetworkEnsemble<double> >( n )
{
}

vector<HMAXInterface::value_type> HMAXInterface::feature_vector()
{
	vector<value_type> r;

	int lastlayer = _network.begin()->nr_layers() - 2;
	for( Network::network::iterator i = _network.begin()->begin( lastlayer );
		i != _network.begin()->end( lastlayer );
		i++ )
	{
		r.push_back( *(i)->activation().begin() );
	}

	return r;
}
