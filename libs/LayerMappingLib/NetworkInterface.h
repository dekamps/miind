#ifndef LAYERMAPPINGLIB_NETWORKINTERFACE_H
#define LAYERMAPPINGLIB_NETWORKINTERFACE_H

#include <vector>
#include <algorithm>
#include <map>

#include "algorithm.h"

using namespace std;

namespace LayerMappingLib
{
	template<class Network>
	class NetworkInterface
	{
		public:
		typedef typename Network::value_type value_type;

		NetworkInterface( Network n );
	
		Network& network();

		/*! \brief Assign image to input node

			This is the default function for setting the input activations. It is assumed that the node is the first node in the first network of an network ensemble.

			\param image The input image */
		void set_input( const std::vector<double>& image );
		/*! \brief Assign image to input node

			This is the default function for setting the input activations. It is assumed that the node is the first node in the first network of an network ensemble.

			\param image The input image */
		void set_input( double* image );

		/*! \brief Evolve the network */
		void evolve();

		/*! \brief Evovle the i-th network in an Ensemble */
		void evolve( int i );

		void set_feedback_template( double* feedback_template );
		void set_feedback_template( const std::vector<double>& feedback_template );

		vector<double> feature_vector();
		vector<double> feature_vector( int n );

		vector<double> feature_map( int network, int layer, int feature );
		pair<double, double> feature_map_size( int nework, int layer, int feature );

		int nr_feature_maps( int network, int layer );

		void fill_feature_map_padding_with_noise( double level );

		protected:
		Network _network;
	};
}
#endif //LAYERMAPPINGLIB_NETWORKINTERFACE_H
