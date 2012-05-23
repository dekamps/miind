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

#ifndef LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODE_H
#define LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODE_H

#include <vector>
#include <map>

#include "ReceptiveFieldCode.h"
#include "FeatureMapCode.h"
#include "AbstractFunction.h"

namespace LayerMappingLib
{
	/*! \class FeatureMapNode
		\brief A node, whose activation is a FeatureMap.

		A FeatureMap can be regarded as a layer of nodes. A FeatureMapNode is a generalisation of a node. Where the activation of a node is a single value, the activation of a FeatureMapNode is a FeatureMap. 

		Each node in that FeatureMap has a receptive field assigned to it. Receptive field size and skipping are identical for each node in the FeatureMap. When the FeatureMapNode is evaluated each node in the FeatureMap is eveluated.

		The activation and predecessors can be accessed by iterator. */
	template<class T>
	class FeatureMapNetwork;

	template<class T, class F>
	class FeatureMapNode
	{
		friend class FeatureMapNetwork<T>;

		public:
		typedef ReceptiveField<T> receptive_field;
		typedef typename receptive_field::iterator receptive_field_iterator;
		typedef typename FeatureMap<T>::iterator iterator;

		typedef typename F::vector_list vector_list;

		/*! \brief Constructor 

			Construct a FeatureMapNode with predecessors p and activation a. The receptive field size and the skipping are determined by x_skip, y_skip, rf_width and rf_height. The padding size, determined by padding_width and padding_height specifies how much padding from the predecessors is taken into account.

			\param p A vector with pointers to the nodes predecessors.
			\param a The activaton of the FeatureMapNode.
			\param x_skip Skips in x.
			\param x_skip Skips in y.
			\param input_rf_width The width of the receptive fields in input.
			\param input_rf_height The height of the receptive fields in input.
			\param output_rf_width The width of the receptive fields in output.
			\param output_rf_height The height of the receptive fields in output.
			\param input_padding_width The width of the padding in the predecessor layers taken into account.
			\param input_padding_height The height of the padding in the predecessor layers taken into account.*/
		FeatureMapNode( vector<FeatureMapNode<T, F>* >& p,
			FeatureMap<T> a,
			F* function,
			const string& description,
			int x_skip, int y_skip,
			int input_rf_width, int input_rf_height,
			int output_rf_width, int output_rf_height,
			int input_padding_width, int input_padding_height );

		FeatureMapNode();
		FeatureMapNode( const FeatureMapNode<T, F>& n );

		void update();

		FeatureMapNode<T, F>& operator=( const FeatureMapNode<T, F>& n );

		iterator activation_begin();
		iterator activation_end();

		/*! \brief A vector of FeatureMap iterators that points to the first position of the corresponding predecessor.

		The ith entry in the vector of FeatureMap iterators points to the first position of the activation of the ith predecessor.*/
		vector<iterator> predecessor_begin();
		/*! \brief A vector of FeatureMap iterators that point past the last position of the corresponding predecessor. */
		vector<iterator> predecessor_end();

		bool has_predecessors() const;
		int nr_predecessors() const;

		FeatureMap<T> activation();
		void set_activation( FeatureMap<T> sl );

		void set_skip_size( int width, int height );
		void set_receptive_field_size( int width, int height );
		void set_input_padding_size( int width, int height );
		void set_function( F* function );
		void set_description( const string& description );
		void set_output_receptive_field_size( int width, int height );

		/*! \brief Add predecessors to the node. */
		void add_predecessors( vector<FeatureMapNode<T, F>* >& p );

		const string description() const;
		int width() const;
		int height () const;
		int recpetive_field_witdh() const;
		int receptive_field_height() const;
		int x_skip() const;
		int y_skip() const;
		int output_rf_width() const;
		int output_rf_height() const;
		int input_padding_widht() const;
		int input_padding_height() const;
		F* function() const;

		#ifdef DEBUG
		void debug_print();
		#endif //DEBUG

		private:
		FeatureMap<T> _activation;
		vector<FeatureMapNode<T, F>* > _v_p_predecessor;

		F* _function;

		string _description;

		int _x_skip;
		int _y_skip;

		int _input_rf_width;
		int _input_rf_height;

		int _output_rf_width;
		int _output_rf_height;

		int _input_padding_width;
		int _input_padding_height;
	};
}

#endif //LAYERMAPPINGIMPLEMENTATIONLIB_FEATUREMAPNODE_H
