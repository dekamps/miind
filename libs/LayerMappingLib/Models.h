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

#ifndef LAYERMAPPINGLIB_MODELS_H
#define LAYERMAPPINGLIB_MODELS_H

#include <math.h>

#include "Util.h"

#include "NetworkEnsembleCode.h"
#include "FeatureMapNetworkCode.h"
#include "FunctionFactoryCode.h"
#include "NetworkInterfaceCode.h"
// #include "HMAXInterface.h"

/*! \ingroup LayerMappingLib

	\page LayerMappingLib LayerMappingLib

	\example LayerMappingLib Examples can be found in \ref Models.cpp

*/
using namespace std;

namespace LayerMappingLib
{
	/*! \class Models
		\brief Implementations of various hierarchical models.
		*/
	struct Models
	{
		public:
		typedef double value_type;
		typedef FeatureMapNetwork<value_type> network;
		typedef NetworkEnsemble<value_type> network_ensemble;
		typedef FeatureMapNetwork<value_type>::function_factory function_factory;
		typedef FeatureMapNetwork<value_type>::function function;
		typedef FeatureMapNetwork<value_type>::node node;
		
		typedef pair<vector<string>, vector<function*> > pair_description_function;
		/*! \brief For testing

			Performs a Min and Min in the first layer, Mean in the second layer. The third perfoms an ArgMax */
		static network SimpleTest( int width, int height );
		/*! \brief Simple feedback example */
		static NetworkInterface<network_ensemble> SimpleFeedback( int width, int height );
		/*! \brief Check if the connections between networks work.*/
		static NetworkInterface<network_ensemble> SimpleEnsemble( int width, int height );
		/*! \brief Implementation of the original HMAX model by Tomaso Poggio and Maximilian Riesenhuber, as described in "Hierarchical Models of Object Recognition in Cortex". */
		static NetworkInterface<network_ensemble> HMAX( int width, int height, const string& C1_pooling_operation, const string& C2_pooling_operation/*, const vector<vector<double> >& weights, const vector<string>& object_names*/ );
		/*! \brief Modified HMAX and feedback network. */
		static NetworkInterface<network_ensemble> HMAX_Feedback( int width, int height, const string& C1_pooling_operation, const string& C2_pooling_operation /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ );
		/*! \brief Implementation of the extension of the HMAX model by Thomas Serre, as described in "Robust Object Recognition with Cortex-Like Mechanisms" */
		static NetworkInterface<network_ensemble> HMAX_Learned_S2( int width, int height,
			const vector<int>& filter_bands,
			const vector<string>& S1_features,
			const string& C1_pooling_operation, const vector<int>& C1_receptive_field_size, const vector<int>& C1_skip_size,
			const vector<string> S2_features,
			const string& C2_pooling_operation /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ );
		/*! \brief Implementation of the saliency model of Itti et al. as described in "Modeling attention to salient proto-objects" by Dirk Walther and Christof Koch. */
//		static network Saliency( int width, int height ); //TODO

		static NetworkInterface<network_ensemble> ConvolutionTest( int width, int height, int kernel_width, int kernel_height );
		
		static  pair<vector<string>, vector<function*> > functions_from_strings( const vector<string>& function_strings );
		/*! \brief determine the padding size with respect to filters. */
		static pair<int, int> padding_size( const vector<function*>& functions );
		private:
		static network _HMAX( int width, int height, const vector<int>& filter_bands, const string& C1_function, const string& C2_function/*, const vector<vector<double> >& weights, const vector<string>& object_names*/ );
		static network _HMAX_Learned_S2( int width, int height,
			const vector<int>& filter_bands, const vector<string> S1_features,
			const string& C1_function, const vector<int>& C1_receptive_field_size, const vector<int>& C1_skip_size,
			const vector<string> S2_features, const string& C2_function /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ );

		static vector<node> _add_signalpath( FeatureMapNetwork<value_type>& mapping_network, 
			int signal_path,
			int layer,
			vector<vector<node > >& predecessors,
			vector<function*> functions,
			int output_padding_width, int output_padding_height,
			int inut_skip_width, int input_skip_height,
			int output_skip_width, int output_skip_height,
			vector<int>& receptive_field_width, vector<int>& receptive_field_height,
			const string& description );

		static void _centered_filter_matrix( int width, int height, int rowstride, vector<vector<double> >& v, double* r );
		static void _centered_filter_matrix_upper_left_corner( int width, int height, int rowstride, vector<vector<double> >& v, double* r  );

	};
}

#endif //LAYERMAPPINGLIB_MODELS_H
