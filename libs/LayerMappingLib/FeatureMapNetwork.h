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

#ifndef LAYERMAPPINGLIB_FEATUREMAPNETWORK_H
#define LAYERMAPPINGLIB_FEATUREMAPNETWORK_H

#include <algorithm>
#include <vector>

#include "boost/lambda/lambda.hpp"
#include "boost/lambda/bind.hpp"
#include "boost/lambda/construct.hpp"

#include "algorithm.h"
#include "FeatureMapNodeCode.h"
#include "AbstractFunction.h"
#include "FunctionFactoryCode.h"

#include "../UtilLib/UtilLib.h"

using namespace std;

namespace LayerMappingLib
{
	/*! \class FeatureMapNetwork
		\brief A network of FeatureMapNodes.

			Note: An input node is per definition a node, that has no predecessors.*/
	template<class T>
	class FeatureMapNetwork
	{
		public:
		typedef vector<ReceptiveField<T> > vector_list;
		typedef AbstractFunction<vector_list> function;
		typedef FunctionFactory<function> function_factory;

		typedef FeatureMapNode<T, function> node;

		typedef typename UtilLib::PointerContainerIterator<typename vector<FeatureMapNode<T, function>* >::iterator> iterator;

		/*! \brief Standard Constructor

			This is required by swig. Do not use this constructor */
		FeatureMapNetwork();
		/*! \brief Constructor
			\param nr_layer The number of layers in the network has to be specified at instatiation.
			\param title Give your network a name.*/
		FeatureMapNetwork( int nr_layer, const string& title );
		FeatureMapNetwork( const FeatureMapNetwork<T>& );
		/*! \brief Construct an inhibition network out of two networks.

			The activation of a node is calculated as the product of two nodes at the same position. The two networks have to have the same structure. */
		FeatureMapNetwork( /*const TODO*/ FeatureMapNetwork<T>& n1, /*const TODO*/ FeatureMapNetwork<T>& n2, const string& title );

		~FeatureMapNetwork();

		FeatureMapNetwork<T>& operator=( const FeatureMapNetwork<T>& );
		
		/*! \brief Generate a network with the same architecture.*/
		FeatureMapNetwork<T> clone( const string& title );

		/*! \brief Add a FeatureMapNode that has no predecessors (and hence is an input node)
			\param activation The activation of the FeatureMapNode to be added.
			\param node_description The description of the FeatureMapNode to be added.
			\param layer_number Specifies the layer where the node is added. */
		node* add_node( FeatureMap<T> activation, int layer = 0, const string& node_description = "input" );
		/*! \brief Add a FeatureMapNode.
			\param activation The activation of the FeatureMapNode to be added.
			\param f The function assigned to the FeatureMapNode to be added.
			\param node_description The description of the FeatureMapNode to be added.
			\param layer_number Specifies the layer where the node is added.*/
		node* add_node( vector<FeatureMapNode<T, function>* >& predecessors,
			FeatureMap<T> activation,
			function* f,
			int layer_number,
			int x_skip, int y_skip,
			int input_rf_width, int input_rf_height,
			int input_padding_width, int input_padding_height,
			int output_rf_width, int output_rf_height,
			const string& node_description );

		/*! \brief Add a layer with simple cell feature maps. The number of the features in the feature dictionary is the length of the functions. For each feature map the of layer defined by predecessor_layer_number new feature maps are added according to the number of feature in the dictionary.
			\param predecessor_layer_number the layer with the preceding nodes
			\param functions Functions defining how the feature maps are evolved. This is typically a feature dictionary.
			\param padding_width The width of the padding
			\param padding_height The height of the padding */
		vector<node*> add_simple_cell_layer( int preceding_layer_number, int layer_number, const vector<function*>& functions, const vector<string>& descriptions, vector<int>& padding_width, vector<int>& padding_height );

		vector<node*> add_complex_cell_layer( int preceding_layer_number, int layer_number,
			const vector<function*> pooling_functions, const vector<string>& descriptions,
			int offset,
			const vector<int>& filter_bands,
			const vector<int>& receptive_field_width, const vector<int>& receptive_field_height,
			const vector<int>& skip_width, const vector<int>& skip_height,
			const vector<int>& padding_width, const vector<int>& padding_height );

		/*! \brief Iterator, that points to the first node in the network. */
		iterator begin();
		/*! \brief Iterator, that points past the last node in the network. */
		iterator end();

		/*! \brief Iterator, that points to the first node in the ith layer.
			\param i Specifies the layer to iterate through. */
		iterator begin( int i );
		/*! \brief Iterator, that points past the last node in the network ith layer.
			\param i Specifies the layer to iterate through. */
		iterator end( int i );

		iterator node_at( int layer, int i );

		/*! \brief Make the outputs of the network n inputs of this network

			\param n The network whose ouputs are made the input of this network.*/
		void connect( /*const TODO*/ FeatureMapNetwork<T>& n );

		void set_layer_name( int i, const string& s );
		const string& layer_name( int i ) const;

		int nr_layers() const;
		int nr_feature_maps( int i ) const;

		/*! \brief The activation of all input nodes.*/
		vector<FeatureMap<T> > layer_activation( int i = 0);

		/*! \brief Update the network
			Update every node in the network. The order is defined by the order of insertion of nodes. Nodes that are added first are updated first.*/
		void update();

		const string title() const;

		#ifdef DEBUG
		void debug_print();
		#endif //DEBUG

		void fill_feature_map_padding_with_noise( double level );

		private:
		vector<node*> _clone_nodes( const FeatureMapNetwork<T>& n );

		vector<node*> _v_node;

		vector<int> _v_layer_size;

		vector<string> _v_layer_name;

		string _title;
	};
}

#endif //LAYERMAPPINGLIB_FEATUREMAPNETWORK_H

