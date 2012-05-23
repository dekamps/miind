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

#include "Models.h"

using namespace LayerMappingLib;

NetworkInterface<Models::network_ensemble> Models::SimpleEnsemble( int width, int height )
{
	network_ensemble ensemble;

	network n1( 2, "Forward Network" );
	network n2( 2, "Backward Network" );

	//network 1
	{
		string function_name = PRODUCT;
		stringstream ss( stringstream::in | stringstream::out );
		ss << function_name;

		vector<node*> layer0;

		layer0.push_back( n1.add_node( FeatureMap<value_type>( width, height, 0, 0 ) ) );
		layer0.push_back( n1.add_node( FeatureMap<value_type>( width, height, 0, 0 ), 0, "Inhibitor" ) );

		n1.add_node( layer0,
			FeatureMap<value_type>( width, height, 0, 0 ),
			function_factory::get_function( ss ),
			1,
			1, 1,
			1, 1,
			0, 0,
			1, 1,
			function_name );
	}
	//network 2
	{
		string function_name = ARGMAX;
		stringstream ss( stringstream::in | stringstream::out );
		ss << function_name;

		vector<node*> layer0;

		FeatureMap<value_type> n1_output = n1.begin( 1 )->activation();

		layer0.push_back( n2.add_node( n1_output, 0, "n1_ouput" ) );

		vector<node*> layer1;
		layer1.push_back( n2.add_node( layer0,
			FeatureMap<value_type>( width, height, 0, 0 ),
			function_factory::get_function( ss ),
			1,
			2, 2,
			2, 2,
			0, 0,
			2, 2,
			function_name ) );

		(++n1.begin( 0 ))->set_activation( n2.begin( 1 )->activation() );

	}

	node inhibitor = *(++n1.begin( 0 ));
// 	generate( inhibitor.activation().begin(), inhibitor.activation().end(), (double) rand() / RAND_MAX + _1 );
	for( FeatureMap<value_type>::iterator i = inhibitor.activation().begin();
		i != inhibitor.activation().end();
		i++ )
	{
		*i = (double) rand() / RAND_MAX;
	}


	ensemble.add_network( n1 );
	ensemble.add_network( n2 );

	return NetworkInterface<network_ensemble>( ensemble );
}

Models::network Models::SimpleTest( int width, int height )
{
	network layer_mapping_network( 4, "SimpleTest" );

	stringstream ss( stringstream::in | stringstream::out );
	ss <<
	//layer 1
		MIN << " " <<
		MAX << " " <<
	//layer 2
		MEAN << " " <<
	//layer 3
		ARGMAX;

//--------------layer 0
	vector<node*> layer0;
	layer0.push_back( layer_mapping_network.add_node( FeatureMap<value_type>( width, height, 0, 0 ) ) );

	generate( layer_mapping_network.layer_activation( 0 ).front().begin(), 
		layer_mapping_network.layer_activation( 0 ).front().end(),
		incrementer<value_type>( 0, 1 ) );

//-----------------layer 1
	vector<node*> layer1;
	layer1.push_back( layer_mapping_network.add_node( layer0,
		FeatureMap<value_type>( width / 2, height / 2, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		2, 2,
		2, 2,
		0, 0,
		1, 1,
		"min" ) );

	layer1.push_back( layer_mapping_network.add_node( layer0,
		FeatureMap<value_type>( width / 2, height / 2, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		2, 2,
		2, 2,
		0, 0,
		1, 1,
		"max" ) );

//------------------layer 2
	vector<node*> layer2;
	layer2.push_back( layer_mapping_network.add_node( layer1,
		FeatureMap<value_type>( width / 4, height / 4, 0, 0 ),
		function_factory::get_function( ss ),
		2,
		2, 2,
		2, 2,
		0, 0,
		1, 1,
		"mean" ) );

//------------------layer 2
	vector<node*> layer3;
	layer3.push_back( layer_mapping_network.add_node( layer2,
		FeatureMap<value_type>( width / 4, height / 4, 0, 0 ),
		function_factory::get_function( ss ),
		3,
		2, 2,
		2, 2,
		0, 0,
		2, 2,
		"argmax" ) );

	return layer_mapping_network;
}

NetworkInterface<Models::network_ensemble> Models::SimpleFeedback( int width, int height )
{
	//This does not work any more!
	network_ensemble ensemble;

	network layer_mapping_network = SimpleTest( width, height );

	ensemble.add_network( layer_mapping_network );
	
	network feedback = layer_mapping_network.clone( "Feedback" );
// 	feedback.connect( layer_mapping_network );

	ensemble.add_network( feedback );

	network local_consistency_check( layer_mapping_network, feedback, "Local consitency check" );
	ensemble.add_network( local_consistency_check );

	return NetworkInterface<network_ensemble>( ensemble );
}

NetworkInterface<Models::network_ensemble> Models::ConvolutionTest( int width, int height, int kernel_width, int kernel_height )
{
	network_ensemble ne;

	#ifndef HAVE_FFTW
	network layer_mapping_network( 2, "ConvolutionTest" );

	stringstream ss( stringstream::in | stringstream::out );
	ss <<
	//layer 1
		CONVOLUTION << " " << kernel_width << " " << kernel_height << " ";
// 	vector<vector<double> > d = gabor( kernel_height, kernel_width, 2.5, 3.0, 135.0, 0.3 );
	vector<vector<double> > d = second_derivat_gaussian( 0.3, kernel_height, kernel_width, 135.0 );
	for( vector<vector<double> >::iterator i = d.begin();
		i != d.end();
		i++ )
	{
		copy( i->begin(), i->end(), ostream_iterator<double>( ss, " " ) );
	}
// 	for( int i = 0; i < kernel_width * kernel_height; i++ )
// 	{
// 		ss << "1 ";
// 	}

//--------------layer 0
	vector<node*> layer0;
	layer0.push_back( layer_mapping_network.add_node( FeatureMap<value_type>( width, height, kernel_width - 1, kernel_height - 1 ) ) );

//-----------------layer 1
	vector<node*> layer1;
	layer1.push_back( layer_mapping_network.add_node( layer0,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		1, 1,
		kernel_width, kernel_height,
		kernel_width - 1, kernel_height - 1,
		1, 1,
		"convolution" ) );

	#else //HAVE_FFTW
	network layer_mapping_network( 5, "ConvolutionTest" );
	layer_mapping_network.set_layer_name( 0, "Input" );
	layer_mapping_network.set_layer_name( 1, "Fourier Transform" );
	layer_mapping_network.set_layer_name( 2, "Check" );
	layer_mapping_network.set_layer_name( 3, "Product" );
	layer_mapping_network.set_layer_name( 4, "Output" );

	stringstream ss( stringstream::in | stringstream::out );
	ss << 
		//layer 1
		FFTR_ << " " << FFT_TYPE_REAL << " " << FFT_FORWARD << " " << width << " " << height << " " <<
		FFTR_ << " " << FFT_TYPE_IMAG << " " << FFT_FORWARD << " " << width << " " << height << " " <<
		FFTR_ << " " << FFT_TYPE_REAL << " " << FFT_FORWARD << " " << width << " " << height << " " <<
		FFTR_ << " " << FFT_TYPE_IMAG << " " << FFT_FORWARD << " " << width << " " << height << " " <<
		//layer 2
		FFTC_ << " " << FFT_TYPE_REAL << " " << FFT_BACKWARD << " " << width << " " << height << " " <<
		FFTC_ << " " << FFT_TYPE_IMAG << " " << FFT_BACKWARD << " " << width << " " << height << " " <<
		FFTC_ << " " << FFT_TYPE_REAL << " " << FFT_BACKWARD << " " << width << " " << height << " " <<
		FFTC_ << " " << FFT_TYPE_IMAG << " " << FFT_BACKWARD << " " << width << " " << height << " " <<
		//layer 3
		PRODUCT_COMPLEX_REAL << " " <<
		PRODUCT_COMPLEX_IMAG << " " <<
		//layer 4
		FFTC_ << " " << FFT_TYPE_REAL << " " << FFT_BACKWARD << " " << width << " " << height << " " <<
		FFTC_ << " " << FFT_TYPE_IMAG << " " << FFT_BACKWARD << " " << width << " " << height;

	//--------------layer 0
	vector<node*> layer0_image;
	vector<node*> layer0_filter;
	layer0_image.push_back( layer_mapping_network.add_node( FeatureMap<value_type>( width, height, 0, 0 ) ) );
	layer0_filter.push_back( layer_mapping_network.add_node( FeatureMap<value_type>( width, height, 0, 0 ) ) );

// 	vector<vector<double> > d = gabor( kernel_height, kernel_width, 2.5, 3.0, 135.0, 0.3 );
// // 	vector<vector<double> > d = gaussian( kernel_height, kernel_width, 1.0 );	
// // 	vector<vector<double> > d = second_derivat_gaussian( 0.8, kernel_width, kernel_height, 45 );
// 	double cf[ width * height ];
// 	for( vector<vector<double> >::iterator i = d.begin();
// 		i != d.end();
// 		i++ )
// 	{
// 		fill( i->begin(), i->end(), 1.0 / width * height );
// 	}
// // 	_centered_filter_matrix( width, height, width, d, cf );
// 	_centered_filter_matrix_upper_left_corner( width, height, width, d, cf );
// 
	double cf[ width * height ];
	fill( cf, cf + width * height, 0 );
	*cf = 1;
	layer0_filter.front()->activation().get( cf );



	//-----------------layer 1
	vector<node*> layer1_image;
	vector<node*> layer1_filter;
	layer1_image.push_back( layer_mapping_network.add_node( layer0_image,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Image FFT Real" ) );

	layer1_image.push_back( layer_mapping_network.add_node( layer0_image,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filter FFT Imag" ) );

	layer1_filter.push_back( layer_mapping_network.add_node( layer0_filter,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filter FFT Real" ) );

	layer1_filter.push_back( layer_mapping_network.add_node( layer0_filter,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		1,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filter FFT Imag" ) );

	//--------------------------layer 2
	vector<node*> layer2_image;
	vector<node*> layer2_filter;
	layer2_image.push_back( layer_mapping_network.add_node( layer1_image,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		2,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Image Real" ) );

	layer2_image.push_back( layer_mapping_network.add_node( layer1_image,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		2,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Image Imag" ) );

	layer2_filter.push_back( layer_mapping_network.add_node( layer1_filter,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		2,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filter Real" ) );

	layer2_filter.push_back( layer_mapping_network.add_node( layer1_filter,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		2,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filter Imag" ) );

	//--------------layer 3
	vector<node*> layer3;
	vector<node*> t( layer1_image );
	t.push_back( layer1_filter.at( 0 ) );
	t.push_back( layer1_filter.at( 1 ) );

	layer3.push_back( layer_mapping_network.add_node( t,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		3,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filtered Image FFT Real" ) );

	layer3.push_back( layer_mapping_network.add_node( t,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		3,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filtered Image FFT Imag" ) );

	//--------------layer 3
	layer_mapping_network.add_node( layer3,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		4,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filtered Image Real" );
	layer_mapping_network.add_node( layer3,
		FeatureMap<value_type>( width, height, 0, 0 ),
		function_factory::get_function( ss ),
		4,
		width, height,
		width, height,
		0, 0,
		width, height,
		"Filtered Image Imag" );
	#endif //HAVE_FFTW

	ne.add_network( layer_mapping_network );
	
	return NetworkInterface<network_ensemble>( ne );
}

// 
// // FeatureMapNetwork<Models::T> Models::Saliency( int width, int height )
// // {
// // 
// // 	typedef network::function_factory function_factory;
// // 	typedef network::function function;
// // 
// // 	network layer_mapping_network( 3 );
// // 
// // 	layer_mapping_network.set_layer_name( 0, "Input" );
// // 	layer_mapping_network.set_layer_name( 1, "Dyadic Gaussian Pyramid" );
// // 	layer_mapping_network.set_layer_name( 2, "Intensity Map" );
// // 
// // 	int input_padin_width = 0; //TODO allow padding for input sizes, that are not powers of two
// // 	int input_padin_height = 0;
// // 
// // //----------------layer 0
// // 	vector<node*> layer0;
// // 	layer0.push_back(
// // 		layer_mapping_network.add_node(
// // 			layer_mapping_network.add_FeatureMap( width, height, 
// // 					input_padin_width, input_padin_height ),
// // 			"red" ) );
// // 	layer0.push_back(
// // 		layer_mapping_network.add_node(
// // 			layer_mapping_network.add_FeatureMap( width, height, 
// // 					input_padin_width, input_padin_height ),
// // 			"green" ) );
// // 	layer0.push_back(
// // 		layer_mapping_network.add_node(
// // 			layer_mapping_network.add_FeatureMap( width, height, 
// // 					input_padin_width, input_padin_height ),
// // 			"blue" ) );
// // 
// // //----------------------layer 1
// // 	vector<node*> layer1;
// // 
// // 	return layer_mapping_network;
// // }
// 

Models::network Models::_HMAX( int width, int height, const vector<int>& filter_bands, const string& C1_function, const string& c2_function/*, const vector<vector<double> >& weights, const vector<string>& object_names*/ )
{
	//What is happening here:
	//	*some global definitions
	//	*size, padding, functions, etc are defined for each layer from S1 to c2
	//	*most of the variables are printed to std out for debugging
	//	*each layer is initialised

	//global definitions
	int nr_layer = 6;

	network network( nr_layer, "HMAX" );
	network.set_layer_name( 0, "input" );
	network.set_layer_name( 1, "S1" );
	network.set_layer_name( 2, "C1" );
	network.set_layer_name( 3, "S2" );
	network.set_layer_name( 4, "C2'" );
	network.set_layer_name( 5, "C2" );
// 	network.set_layer_name( 6, "C2''" );
// 	network.set_layer_name( 7, "VTU" );

	int filtersize_inc = 2;

	int nr_orientation = 4;
	vector<double> orientation( nr_orientation );
	double deg = 0;
	for( int i = 0; i < nr_orientation; i++ )
	{
		orientation.at( i ) = deg;
		deg += ( 180.0 / nr_orientation );
	}

	int nr_signalpath = filter_bands.size();
	int nr_function_S1 = 0;
	for( vector<int>::const_iterator i = filter_bands.begin();
		i != filter_bands.end();
		i++ )
	{
		nr_function_S1 += *i * nr_orientation;
	}

	double sigDivisor = 4.0;
	
	//C1
	vector<int> C1_rf_width;
	vector<int> C1_rf_height;
	C1_rf_width.push_back( 4 ); C1_rf_height.push_back( 4 );
	C1_rf_width.push_back( 6 ); C1_rf_height.push_back( 6 );
	C1_rf_width.push_back( 9 ); C1_rf_height.push_back( 9 );
	C1_rf_width.push_back( 12 ); C1_rf_height.push_back( 12 );

	double C1_overlay = 2.0;

	//s2
	int s2_receptivefield_size = 2;
	int composite_nr_positions = 4;

	//VTU
// 	assert( weights.size() == object_names.size() );
// 	for( vector<vector<double> >::const_iterator i = weights.begin();
// 		i != weights.end();
// 		i++ )
// 	{
// // 		assert( i->size() == std::pow( composite_nr_positions,  s2_receptivefield_size * s2_receptivefield_size ) );
// 	}
// 	int nr_objects = object_names.size();

	#ifdef DEBUG
	cout << "Global variables..." << endl;
	cout << "# of signalpaths: " << nr_signalpath << endl;
	cout << "# of orientations: " << nr_orientation << endl;
	cout << "# of filters per signalpath in S1: ";
	copy( filter_bands.begin(), filter_bands.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
	cout << "# of all filters in S1: " << nr_function_S1 << endl;
	cout << "# of nodes in C1: " << nr_orientation * nr_signalpath << endl;
	cout << "# of nodes in s2: " << pow( (double) composite_nr_positions, nr_orientation ) << endl;
	cout << "C1 receptive field width:\t";
	copy( C1_rf_width.begin(), C1_rf_width.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
	cout << "C1 receptive field height:\t";
	copy( C1_rf_height.begin(), C1_rf_height.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
	cout << "C1 receptive field overlay: " << C1_overlay << endl;
	cout << "s2 receptive field size: " << s2_receptivefield_size << endl << " # composite positions: " << composite_nr_positions << endl;
	#endif //DEBUG

	//layer definitions
	vector<function*> S1_functions;
	vector<int> S1_rf_width;
	vector<int> S1_rf_height;

	int S1_input_x_skip = 1;
	int S1_input_y_skip = 1;

	int S1_output_x_skip = 1;
	int S1_output_y_skip = 1;

	int S1_width = width / S1_output_x_skip;
	int S1_height = height / S1_output_y_skip;

	int input_padding_width = 0;
	int input_padding_height = 0;
	{
		int fs = 7;
		for( int i = 0; i < nr_signalpath; i++ )
		{
			for( int k = 0; k < nr_orientation; k++ )
			{
				int t_fs = fs;
				for( int j = 0; j < filter_bands.at( i ); j++ )
				{	
					S1_rf_width.push_back( t_fs );
					S1_rf_height.push_back( t_fs );
			
					input_padding_width = max( input_padding_width, t_fs - 1 );
					input_padding_height = max( input_padding_height, t_fs - 1 );

					vector<vector<double> > filter = second_derivat_gaussian( sigDivisor,
						t_fs, t_fs,
						orientation.at( k ) );	

					stringstream ss( stringstream::in | stringstream::out );

					ss << CONVOLUTION << " "<< t_fs << " " << t_fs << " ";
					for( vector<vector<double> >::iterator b = filter.begin();
						b != filter.end();
						b++ )
					{
						copy( b->begin(), b->end(), ostream_iterator<double>( ss, " " ) );
					}
					S1_functions.push_back( function_factory::get_function( ss ) );

					t_fs += filtersize_inc;
				}
			}
			fs += filtersize_inc * filter_bands.at( i );
		}
	}

	//-------------determine S1 padding and C1 size and skip size
	vector<int> S1_padding_width( nr_signalpath );
	vector<int> S1_padding_height( nr_signalpath );

	vector<int> C1_input_x_skip( nr_signalpath );
	vector<int> C1_input_y_skip( nr_signalpath );

	vector<int> C1_width( nr_signalpath );
	vector<int> C1_height( nr_signalpath );

	vector<int> C1_padding_width( nr_signalpath );
	vector<int> C1_padding_height( nr_signalpath );

	for( int i = 0; i < nr_signalpath; i++ )
	{
		C1_input_x_skip.at( i ) = (int) ceil( C1_rf_width.at( i ) / C1_overlay );
		C1_input_y_skip.at( i ) = (int) ceil( C1_rf_height.at( i ) / C1_overlay );

 		S1_padding_width.at( i ) = (int) ceil( (double) ( C1_rf_width.at( i ) - 1 ) / 2.0 ) * 2;//(int) ( 2 * ceil( (double) ( ( C1_rf_width.at( i ) - ( S1_width % C1_rf_width.at( i ) ) ) % C1_rf_width.at( i ) ) / 2 ) );
 		S1_padding_height.at( i ) = (int) ceil( ( (double) C1_rf_height.at( i ) - 1 ) / 2.0 ) * 2;//(int) ( 2 * ceil( (double) ( ( C1_rf_height.at( i ) - ( S1_height % C1_rf_height.at( i ) ) ) % C1_rf_height.at( i ) ) / 2 ) );

		C1_width.at( i ) = (int) ceil( (double) S1_width / (double) C1_input_x_skip.at( i ) );
		C1_height.at( i ) = (int) ceil( (double) S1_height / (double) C1_input_y_skip.at( i ) );

		C1_padding_width.at( i ) = s2_receptivefield_size; //TODO is this correct for every rf size?
		C1_padding_height.at( i ) = s2_receptivefield_size;
	}

	//-----------------------s2
	vector<vector<int> > s2_predecessor = combinations( nr_orientation, composite_nr_positions );
	int nr_combinations = s2_predecessor.size();

//	int s2_skip = 1;

	vector<int> s2_width( nr_signalpath );
	vector<int> s2_height( nr_signalpath );
	for( int i = 0; i < nr_signalpath; i++ )
	{
		s2_width.at( i ) = C1_width.at( i );
		s2_height.at( i ) = C1_height.at( i );
	}

//-------------------print layer variables
	#ifdef DEBUG
	cout << "layer variables..." << endl;
	cout << "\tsp\tsize\tpadding\tskip_in\tskip_out\trf" << endl;

	cout << "input" << endl;
	cout << "\t0\t" << width << "x" << height << "\t" << input_padding_width << "x" << input_padding_height << "\t0x0" << "\t0x0" << "\t0x0" << endl;

	cout << "S1" << endl;
	for( int i = 0; i < nr_signalpath; i++ )
	{
		cout << "\t" << i << "\t" << S1_width << "x" << S1_height << "\t" <<
			S1_padding_width.at( i ) << "x" << S1_padding_height.at( i ) << "\t" <<
			S1_input_x_skip << "x" << S1_input_y_skip << "\t" <<
			S1_output_x_skip << "x" << S1_output_y_skip << endl;
	}

	cout << "C1" << endl;
	for( int i = 0; i < nr_signalpath; i++ )
	{
		cout << "\t" << i << "\t" << C1_width.at( i ) << "x" << C1_height.at( i ) << "\t" << C1_padding_width.at( i ) << "x" << C1_padding_height.at( i ) << "\t" << C1_input_x_skip.at( i ) << "x" << C1_input_y_skip.at( i ) << "\t1x1\t" << C1_rf_width.at( i ) << "x" << C1_rf_height.at( i ) << endl;
	}
	cout << "S2" << endl;
	for( int i = 0; i < nr_signalpath; i++ )
	{
		cout << "\t" << i << "\t" << s2_width.at( i ) << "x" << s2_height.at( i ) << "\t0x0\t" << s2_skip << "x" << s2_skip << "\t" << "1x1\t" << s2_receptivefield_size << "x" << s2_receptivefield_size << endl;
	}
// 	cout << "VTUs for: ";
// 	copy( object_names.begin(), object_names.end(), ostream_iterator<string>( cout, " " ) );
// 	cout << endl;
	#endif //DEBUG
//--------------------------------------------------------------------------------------------------
	//initialize layers
	// --------------layer 0
	#ifdef DEBUG
	cout << "init input layer" << endl;
	#endif //DEBUG
	vector<node*> input_layer;
	input_layer.push_back(
		network.add_node(
			FeatureMap<value_type>( width, height, 
					input_padding_width, input_padding_height ) ) );

	// -----------------S1 layer
	#ifdef DEBUG
	cout << "adding S1" << endl;
	#endif //DEBUG
	vector<node*> S1_layer;
	int f = 0;
	for( int i = 0; i < nr_signalpath; i++ )
	{
		for( int k = 0; k < nr_orientation; k++ )
		{
			for( int j = 0; j < filter_bands.at( i ); j++, f++  )
			{
				stringstream ss( stringstream::in | stringstream::out );
				ss << "Gaussian second derivat" << endl << S1_rf_width.at( f ) << "x" << S1_rf_height.at( f ) << endl << orientation.at( k ) << "°";

				S1_layer.push_back( network.add_node( input_layer,
					FeatureMap<value_type>( S1_width, S1_height,
						S1_padding_width.at( i ), S1_padding_height.at( i ) ),
					S1_functions.at( f ),
					1,
					S1_input_x_skip, S1_input_y_skip,
					S1_rf_width.at( f ), S1_rf_height.at( f ),
					S1_rf_width.at( f ) - 1, S1_rf_height.at( f ) - 1,
					S1_output_x_skip, S1_output_y_skip,
					ss.str() ) );
			 }
		}
	}

	//---------------------C1 layer
	#ifdef DEBUG
	cout << "adding C1" << endl;
	#endif //DEBUG
	vector<node*> C1_layer;

	vector<node*>::iterator i_layer = S1_layer.begin();
	for( int current_signalpath = 0;
		current_signalpath < nr_signalpath;
		current_signalpath++ )
	{
		for( int current_orientation = 0;
			current_orientation < nr_orientation;
			current_orientation++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			stringstream info( stringstream::in | stringstream::out );
			ss << C1_function;
			info << " " << current_signalpath << " " << orientation.at( current_orientation ) << "°";

			vector<node*> t_nodes( i_layer, i_layer + filter_bands.at( current_signalpath ) );
			i_layer += filter_bands.at( current_signalpath );

			C1_layer.push_back( network.add_node( t_nodes,
				FeatureMap<value_type>( C1_width.at( current_signalpath ), C1_height.at( current_signalpath ), C1_padding_width.at( current_signalpath ), C1_padding_height.at( current_signalpath ) ),
				function_factory::get_function( ss ),
				2,
				C1_input_x_skip.at( current_signalpath ), C1_input_y_skip.at( current_signalpath ),
				C1_rf_width.at( current_signalpath ), C1_rf_height.at( current_signalpath ),
				S1_padding_width.at( current_signalpath ), S1_padding_height.at( current_signalpath ),
				1, 1,
				info.str() ) );
		}
	}
	assert( i_layer == S1_layer.end() );

	//---------s2
	#ifdef DEBUG
	cout << "adding S2" << endl;
	#endif //DEBUG
	vector<node*> S2_layer;

	for( int i = 0; i < nr_signalpath; i++ )
	{
		vector<node*>::iterator C1_iterator = C1_layer.begin() + i * nr_orientation; //The first feature map in the i-th filter-band
		vector<node*> pred( nr_orientation );
		vector<vector<int> >::iterator i_pred = s2_predecessor.begin();
		for( int j = 0; j < nr_combinations; j++, i_pred++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			stringstream info( stringstream::in | stringstream::out );

			ss << COMPOSITE_FEATURE << " ";

			info << i << "|";
			copy( i_pred->begin(), i_pred->end(), ostream_iterator<int>( info, "" ) );
			info << "|" << j;
			vector<node*>::iterator i_t = pred.begin();
			for( vector<int>::iterator k = i_pred->begin();
				k != i_pred->end();
				k++, i_t++ )
			{
				*i_t = *( C1_iterator + *k );
			}
			assert( i_t == pred.end() );

			S2_layer.push_back( network.add_node( pred,
				FeatureMap<value_type>( s2_width.at( i ), s2_height.at( i ), 0, 0 ),
				function_factory::get_function( ss ),
				3,
				1, 1,
				s2_receptivefield_size, s2_receptivefield_size,
				C1_padding_width.at( i ), C1_padding_height.at( i ),
				1, 1,
				info.str() ) );
		}
		assert( i_pred == s2_predecessor.end() );
	}

	//--------------------c2'
	#ifdef DEBUG
	cout << "adding C2'" << endl;
	#endif //DEBUG
	vector<node*> C2_prime_layer( nr_signalpath * nr_combinations );
	vector<node*>::iterator c = C2_prime_layer.begin();
	for( int i = 0; i < nr_signalpath; i++ )
	{
		vector<node*>::iterator s2_iterator = S2_layer.begin() + i * nr_combinations;
		for( int j = 0; j < nr_combinations; j++, s2_iterator++, c++ )
		{
			vector<node*> pred;

			stringstream ss( stringstream::in | stringstream::out );
			stringstream info( stringstream::in | stringstream::out );
			ss << c2_function;
			info << i << " " << j;

			pred.push_back( *s2_iterator );

			*c = network.add_node( pred,
				FeatureMap<value_type>( 1, 1, 0, 0 ),
				function_factory::get_function( ss ),
				4,
				s2_width.at( i ), s2_height.at( i ),
				s2_width.at( i ), s2_height.at( i ),
				0, 0,
				1, 1,
				info.str() );
		}	
	}
	//--------------------c2
	#ifdef DEBUG
	cout << "adding C2" << endl;
	#endif //DEBUG
	vector<node*> C2_layer( nr_combinations );
	vector<node*>::iterator c2 = C2_layer.begin();
	vector<node*>::iterator s2_iterator = C2_prime_layer.begin();
	for( int j = 0; j < nr_combinations; j++, s2_iterator++, c2++ )
	{
		vector<node*> pred;
		for( int i = 0; i < nr_signalpath; i++ )
		{
			pred.push_back( *( s2_iterator + i * nr_combinations ) );
		}

		stringstream ss( stringstream::in | stringstream::out );
		stringstream info( stringstream::in | stringstream::out );
		ss << c2_function;
		info << j;
		*c2 = network.add_node( pred,
			FeatureMap<value_type>( 1, 1, 0, 0 ),
			function_factory::get_function( ss ),
			5,
			1, 1,
			1, 1,
			0, 0,
			1, 1,
			info.str() );
	}

// 	#ifdef DEBUG
// 	cout << "adding C2''" << endl;
// 	#endif //DEBUG
// 	vector<node*> VTU_layer;
// 	{
// 		stringstream ss( stringstream::in | stringstream::out );
// 		ss << COMBINE;
// 
// 		VTU_layer.push_back( network.add_node( C2_layer,
// 			FeatureMap<value_type>( 1, nr_combinations, 0, 0 ),
// 			function_factory::get_function( ss ),
// 			6,
// 			1, 1,
// 			1, 1,
// 			0, 0,
// 			1, nr_combinations,
// 			"feature vector" ) );
// 	}

// 	//-------------------------vtu
// 	#ifdef DEBUG
// 	cout << "adding VTU" << endl;
// 	#endif
// 	vector<node*> VTU_layer( nr_objects );
// 
// 	for( int i = 0; i < nr_objects; i++ )
// 	{
// 		stringstream ss( stringstream::in | stringstream::out );
// 
// 		ss << PERCEPTRON << " ";
// 		copy( weights.at( i ).begin(), weights.at( i ).end(), ostream_iterator<double>( ss, " " ) );
// 
// 		VTU_layer.push_back( network.add_node( C2_layer,
// 			FeatureMap<value_type>( 1, 1, 0, 0 ),
// 			function_factory::get_function( ss ),
// 			6,
// 			1, 1,
// 			1, 1,
// 			0, 0,
// 			1, 1,
// 			object_names.at( i ) ) );
// 	}

	return network;
}

NetworkInterface<Models::network_ensemble> Models::HMAX( int width, int height,  const string& C1_pooling_operation, const string& c2_pooling_operation /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ )
{
	network_ensemble ensemble;
	
	vector<int> filter_bands;
	filter_bands.push_back( 2 );
	filter_bands.push_back( 3 );
	filter_bands.push_back( 3 );
	filter_bands.push_back( 4 );

	int nr_filter_band = filter_bands.size();
	int s2_size = 256;

	network hmax = ensemble.add_network( Models::_HMAX( width, height,
		filter_bands,
		C1_pooling_operation, c2_pooling_operation/*,
		weights,
		object_names*/ ) );

	network s2_mean( 5, "Statistics" );
	{
		vector<node*> layer0;
		for( network::iterator i = hmax.begin( 3 );
			i != hmax.end( 3 );
			i++ )
		{
			layer0.push_back( s2_mean.add_node( FeatureMap<value_type>( i->activation().width(), i->activation().height(), 0, 0 ) ) );

			layer0.back()->set_activation( i->activation() );
		}
	
		vector<node*> layer1;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MEAN;
			function* f = function_factory::get_function( ss );
			
			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer1.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				1,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"mean" ) );
		}


		vector<node*> layer2;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << STANDARD_DEVIATION;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer2.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				2,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"standard deviation" ) );
		}
		vector<node*> layer3;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MAX;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer3.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				3,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"max" ) );
		}
		vector<node*> layer4;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MIN;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer3.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				4,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"min" ) );
		}

	}

	ensemble.add_network( s2_mean );

	return NetworkInterface<network_ensemble>( ensemble );
}

NetworkInterface<Models::network_ensemble> Models::HMAX_Learned_S2( int width, int height,
			const vector<int>& filter_bands, const vector<string>& S1_features,
			const string& C1_pooling_operation, const vector<int>& C1_receptive_field_size, const vector<int>& C1_skip_size,
			const vector<string> S2_composite_features,
			const string& C2_pooling_operation /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ )
{
	network_ensemble ensemble;
	
	ensemble.add_network( Models::_HMAX_Learned_S2( width, height,
		filter_bands,
		S1_features,
		C1_pooling_operation, 
		C1_receptive_field_size,
		C1_skip_size,
		S2_composite_features,
		C2_pooling_operation/*,
		weights,
		object_names*/ ) );

	return NetworkInterface<network_ensemble>( ensemble );
}

NetworkInterface<Models::network_ensemble> Models::HMAX_Feedback( int width, int height, const string& C1_pooling_operation, const string& c2_pooling_operation /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ )
{
	network_ensemble ne;

	vector<int> filter_bands;
	filter_bands.push_back( 2 );
	filter_bands.push_back( 3 );
	filter_bands.push_back( 3 );
	filter_bands.push_back( 4 );

	int s2_size = 256; //todo get this from network
	int nr_filter_band = filter_bands.size();
	
	network layer_mapping_network = Models::_HMAX( width, height,
		filter_bands,
		C1_pooling_operation, c2_pooling_operation/*,
		weights,
		object_names */);

	ne.add_network( layer_mapping_network );
	
	network feedback = layer_mapping_network.clone( "Feedback" );
	
	//set feedback connections from C2 to C2'
	{
		int l2 = 1;
		int l1 = 0;
		network::iterator t = feedback.begin( l2 );
		for( int i = 0;
			i < nr_filter_band;
			i++ )
		{
			for( network::iterator i = feedback.begin( l1 );
				i != feedback.end( l1 );
				i++, t++ )
			{
				stringstream s( stringstream::in | stringstream::out );
				s << SCALE;
		
				vector<node*> p;
		
				p.push_back( &i );
		
				t->add_predecessors( p );
		
				t->set_skip_size( 0, 0 );
				t->set_receptive_field_size( i->width(), i->height() );
				t->set_output_receptive_field_size( t->width(), t->height() );
				t->set_input_padding_size( 0, 0 );
				t->set_function( function_factory::get_function( s ) );
			}
		}
		assert( t == feedback.end( l2 ) );
	}
// 	feedback.full_correlation_connection( 5, 4 );
	//set feedback connection from C2' to S2
	{
		int l2 = 2;
		int l1 = 1;
		network::iterator t = feedback.begin( l2 );
		for( network::iterator i = feedback.begin( l1 );
			i != feedback.end( l1 );
			i++, t++ )
		{
			stringstream s( stringstream::in | stringstream::out );
			s << SCALE;
	
			vector<node*> p;
	
			p.push_back( &i );
	
			t->add_predecessors( p );
	
			t->set_skip_size( 0, 0 );
			t->set_receptive_field_size( i->width(), i->height() );
			t->set_output_receptive_field_size( t->width(), t->height() );
			t->set_input_padding_size( 0, 0 );
			t->set_function( function_factory::get_function( s ) );
		}
		assert( t == feedback.end( l2 ) );
	}
// 	feedback.full_correlation_connection( 4, 3 );

	ne.add_network( feedback );
// 	feedback.connect( layer_mapping_network );

	network local_consistency_check( layer_mapping_network, feedback, "Local consitency check" );

	ne.add_network( local_consistency_check );

	network s2_mean( 5, "Statistics" );

	{
		vector<node*> layer0;
		for( network::iterator i = local_consistency_check.begin( 7 );
			i != local_consistency_check.end( 7 );
			i++ )
		{
			layer0.push_back( s2_mean.add_node( FeatureMap<value_type>( i->activation().width(), i->activation().height(), 0, 0 ) ) );

			layer0.back()->set_activation( i->activation() );
		}
	
		vector<node*> layer1;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MEAN;
			function* f = function_factory::get_function( ss );
			
			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer1.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				1,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"mean" ) );
		}


		vector<node*> layer2;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << STANDARD_DEVIATION;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer2.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				2,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"standard deviation" ) );
		}
		vector<node*> layer3;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MAX;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer3.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				3,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"max" ) );
		}
		vector<node*> layer4;
		for( int i = 0;	
			i < nr_filter_band;
			i++ )
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss << MIN;
			function* f = function_factory::get_function( ss );

			vector<node*> pred;
			int width = layer0.at( i * s2_size )->activation().width();
			int height = layer0.at( i * s2_size )->activation().height();

			for( int j = 0;
				j < s2_size;
				j++ )
			{
				pred.push_back( layer0.at( j + i * s2_size ) );
			}
			layer3.push_back( s2_mean.add_node( pred,
				FeatureMap<value_type>( width, height, 0, 0 ),
				f,
				4,
				1, 1,
				1, 1,
				0, 0,
				1, 1,
				"min" ) );
		}

	}

	ne.add_network( s2_mean );

	return NetworkInterface<network_ensemble>( ne );
}

void Models::_centered_filter_matrix_upper_left_corner( int width, int height, int rowstride, vector<vector<double> >& v, double* r )
{
	fill( r, r + width * height, 0 );
	
	int mw = v.size();
	int mh = v.front().size();

	//upper left
	double* o = r;
	for( int y = mh / 2;
		y != mh;
		y++, o += rowstride  )
	{
		double* oo = o;
		for( int x = mw / 2;
			x != mw;
			x++, oo++ )
		{
			*oo = v.at( x ).at( y );
		}
	}
	//upper right
	o = r + width - ( mw / 2 );
	for( int y = mh / 2;
		y != mh;
		y++, o += rowstride )
	{
		double* oo = o;
		for( int x = 0;
			x != mw / 2;
			x++, oo++ )
		{
			*oo = v.at( x ).at( y );
		}
	}	
	//lower left
	o = r + ( rowstride * ( height - ( mh / 2 ) ) );
	for( int y = 0;
		y != mh / 2;
		y++, o += rowstride  )
	{
		double* oo = o;
		for( int x = mw / 2;
			x != mw;
			x++, oo++ )
		{
			*oo = v.at( x ).at( y );
		}
	}
	//lower right
	o = r + ( rowstride * ( height - ( mh / 2 ) ) ) + ( width - mw / 2 );
	for( int y = 0;
		y != mh / 2;
		y++, o += rowstride  )
	{
		double* oo = o;
		for( int x = 0;
			x != mw / 2;
			x++, oo++ )
		{
			*oo = v.at( x ).at( y );
		}
	}
}

void Models::_centered_filter_matrix( int width, int height, int rowstride, vector<vector<double> >& v, double* r )
{
	fill( r, r + width * height, 0 );

	int mw = v.front().size();
	int mh = v.size();

	int x = ( width / 2 ) - ( mw / 2 );
	int y = ( height / 2 ) - ( mh / 2 );

	double* p = r + ( ( y * width ) + x );
	for( vector<vector<double> >::iterator row = v.begin();
		row != v.end();
		row++, p += rowstride )
	{
		double* z = p;
		for( vector<double>::iterator c = row->begin();
			c != row->end();
			c++, z++ )
		{
			*z = *c;
		}
	}
}

Models::network Models::_HMAX_Learned_S2( int width, int height,
			const vector<int>& filter_bands,
			const vector<string> S1_features,
			const string& C1_function, const vector<int>& C1_receptive_field_size, const vector<int>& C1_skip_size,
			const vector<string> S2_features, const string& C2_function /*, const vector<vector<double> >& weights, const vector<string>& object_names*/ )
{
	int nr_layer = 6;
	int nr_orientations = 4;

	network network( nr_layer, "HMAX (With learned S2 features)" );
	network.set_layer_name( 0, "input" );
	network.set_layer_name( 1, "S1" );
	network.set_layer_name( 2, "C1" );
	network.set_layer_name( 3, "S2" );
	network.set_layer_name( 4, "C2'" );
	network.set_layer_name( 5, "C2" );

	vector<function*> S1_filter;
	vector<string> S1_descriptions;
	{
		static pair_description_function temp = functions_from_strings( S1_features );

		S1_descriptions = temp.first;
		S1_filter = temp.second;

	}
	vector<function*> S2_filter;
	vector<string> S2_descriptions;
	{
		static pair_description_function temp = functions_from_strings( S2_features );

		S2_descriptions = temp.first;
		S2_filter = temp.second;

	}
	int nr_S2_features = S2_filter.size();

	pair<int, int> input_padding_size = padding_size( S1_filter );

	int nr_filter_bands = filter_bands.size();
	cout << "# of orientations:			" << nr_orientations << endl;
	cout << "# of signalpaths:			" << nr_filter_bands << endl;
	cout << "# of filters per signalpath in S1: 	";
	copy( filter_bands.begin(), filter_bands.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
	cout << "C1 pooling:				" << C1_function << endl;
	cout << "# S2 features:				" << nr_S2_features << endl;
	cout << "C2 pooling:				" << C2_function << endl;


	//initialize layers
	// --------------layer 0
	#ifdef DEBUG
	cout << "init input layer" << endl;
	#endif //DEBUG
	network.add_node(
		FeatureMap<value_type>( width, height, 
				input_padding_size.first, input_padding_size.second ) );

	//--------------------S1
	vector<int> S1_padding_width;
	vector<int> S1_padding_height;

	for( unsigned int i = 0; i < S1_filter.size(); i++ )
	{
		S1_padding_width.push_back( C1_receptive_field_size.at( i / nr_filter_bands ) );
		S1_padding_height.push_back( C1_receptive_field_size.at( i / nr_filter_bands ) );
	}

	network.add_simple_cell_layer( 0, 1,
		S1_filter, S1_descriptions,
		S1_padding_width, S1_padding_height );

	//--------------------C1
	vector<string> C1_pooling;
	vector<function*> C1_pooling_function;
	vector<int> C1_padding;
	int C1_pad;
	{
		pair<int, int> t = padding_size( S2_filter );
		C1_pad = max( t.first, t.second );
	}
	cout << "C1 Padding:				" << C1_pad << endl;
	for( int i = 0;
		i < nr_filter_bands;
		i++ )
	{
		C1_pooling.push_back( C1_function );
		stringstream ss( stringstream::in | stringstream::out );
		ss << C1_function;

		C1_pooling_function.push_back( function_factory::get_function( ss ) );
		C1_padding.push_back( C1_pad );
	}

	network.add_complex_cell_layer( 1, 2,
		C1_pooling_function, C1_pooling,
		nr_orientations,
		filter_bands,
		C1_receptive_field_size, C1_receptive_field_size,
		C1_skip_size, C1_skip_size,
		C1_padding, C1_padding );

	//--------------------S2
	vector<int> S2_padding( nr_orientations * nr_filter_bands * nr_S2_features );
	fill( S2_padding.begin(), S2_padding.end(), 0 );

	vector<node*> S2_layer = network.add_simple_cell_layer( 2, 3,
		S2_filter, S2_descriptions,
		S2_padding, S2_padding );

	//--------------------C2'
	#ifdef DEBUG
	cout << "adding C2'" << endl;
	#endif //DEBUG

	vector<node*> C2_prime_layer( nr_orientations * nr_filter_bands * S2_filter.size() );
	int i = 0;
	cout << "# C2' features " << nr_orientations * nr_filter_bands * S2_filter.size() << endl;
	vector<node*>::iterator S2_iterator = S2_layer.begin();
	for( vector<node*>::iterator c = C2_prime_layer.begin();
		c != C2_prime_layer.end();
		c++, i++, S2_iterator++ )
	{
		vector<node*> pred;

		stringstream ss( stringstream::in | stringstream::out );
		stringstream info( stringstream::in | stringstream::out );
		ss << C2_function;
		info << i;

		pred.push_back( *S2_iterator );

		*c = network.add_node( pred,
			FeatureMap<value_type>( 1, 1, 0, 0 ),
			function_factory::get_function( ss ),
			4,
			(*S2_iterator)->activation().width(), (*S2_iterator)->activation().height(),
			(*S2_iterator)->activation().width(), (*S2_iterator)->activation().height(),
			0, 0,
			1, 1,
			info.str() );
		if( *c == NULL )
		{
			cout << "ja hömma! " << i << endl;
		}
	}
	assert( S2_iterator == S2_layer.end() );

	//------------------C2
	vector<node*> C2_layer( nr_orientations * S2_filter.size() );
	cout << "#C2 features " << nr_orientations * S2_filter.size() << endl;
	vector<node*>::iterator C2_prime_iterator = C2_prime_layer.begin();
	for( unsigned int i = 0;
		i < nr_orientations * S2_filter.size(); 
		i++ )
	{
		vector<node*> pred;

		stringstream ss( stringstream::in | stringstream::out );
		stringstream info( stringstream::in | stringstream::out );
		ss << C2_function;
		info << i;

		for( int j = 0;
			j < nr_filter_bands;
			j++ )
		{
			if( *( C2_prime_iterator + j * nr_orientations * S2_filter.size() + i ) == NULL )
			{
				cout << "aua! " << j * nr_orientations * S2_filter.size() + i << endl;
			}
			pred.push_back( *( C2_prime_iterator + j * nr_orientations * S2_filter.size() + i ) );
		}

		network.add_node( pred,
			FeatureMap<value_type>( 1, 1, 0, 0 ),
			function_factory::get_function( ss ),
			5,
			1, 1,
			1, 1,
			0, 0,
			1, 1,
			info.str() );
	}

// 	network.set_layer_name( 6, "C2''" );
// 	network.set_layer_name( 7, "VTU" );

// 	network network( nr_layer, "HMAX (With learned S2 features)" );
// 	network.set_layer_name( 0, "input" );
// 	network.set_layer_name( 1, "S1" );
// 	network.set_layer_name( 2, "C1" );
// 	network.set_layer_name( 3, "S2" );
// 	network.set_layer_name( 4, "C2'" );
// 	network.set_layer_name( 5, "C2" );
// // 	network.set_layer_name( 6, "C2''" );
// // 	network.set_layer_name( 7, "VTU" );

// 	//What is happening here:
// 	//	*some global definitions
// 	//	*size, padding, functions, etc are defined for each layer from S1 to c2
// 	//	*most of the variables are printed to std out for debugging
// 	//	*each layer is initialised
// 
// 	//global definitions
// 	int nr_layer = 6;
// 
// 	network network( nr_layer, "HMAX (With learned S2 features)" );
// 	network.set_layer_name( 0, "input" );
// 	network.set_layer_name( 1, "S1" );
// 	network.set_layer_name( 2, "C1" );
// 	network.set_layer_name( 3, "S2" );
// 	network.set_layer_name( 4, "C2'" );
// 	network.set_layer_name( 5, "C2" );
// // 	network.set_layer_name( 6, "C2''" );
// // 	network.set_layer_name( 7, "VTU" );
// 
// 	int nr_orientation = 4;
// 	vector<double> orientation( nr_orientation );
// 	double deg = 0;
// 	for( int i = 0; i < nr_orientation; i++ )
// 	{
// 		orientation.at( i ) = deg;
// 		deg += ( 180.0 / nr_orientation );
// 	}
// 
// 	int nr_signalpath = filter_bands.size();
// 	int nr_function_S1 = 0;
// 	for( vector<int>::const_iterator i = filter_bands.begin();
// 		i != filter_bands.end();
// 		i++ )
// 	{
// 		nr_function_S1 += *i * nr_orientation;
// 	}
// 
// 	double sigDivisor = 4.0;
// 	
// 	//s2
// 	int nr_s2_features = s2_composite_features.size();
// 	stringstream ss( stringstream::in | stringstream::out );
// 	ss << s2_composite_features.front();
// 	{
// 		string bla;
// 		ss >> bla;
// 		ss >> bla;
// 	}
// 	ss >> s2_receptive_field_size;
// 
// 
// 	//VTU
// // 	assert( weights.size() == object_names.size() );
// // 	for( vector<vector<double> >::const_iterator i = weights.begin();
// // 		i != weights.end();
// // 		i++ )
// // 	{
// // // 		assert( i->size() == std::pow( composite_nr_positions,  s2_receptivefield_size * s2_receptivefield_size ) );
// // 	}
// // 	int nr_objects = object_names.size();
// 
// 	#ifdef DEBUG
// 	cout << "Global variables..." << endl;
// 	cout << "# of signalpaths: " << nr_signalpath << endl;
// 	cout << "# of orientations: " << nr_orientation << endl;
// 	cout << "# of filters per signalpath in S1: ";
// 	copy( filter_bands.begin(), filter_bands.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "# of all filters in S1: " << nr_function_S1 << endl;
// 	cout << "# of nodes in C1: " << nr_orientation * nr_signalpath << endl;
// 	cout << "# of nodes in s2: " << s2_composite_features.size() << endl;
// 	cout << "C1 receptive field size:\t";
// 	copy( C1_receptive_field_size.begin(), C1_receptive_field_size.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "s2 receptive field size: " << s2_receptive_field_size << endl;
// 	#endif //DEBUG
// 
// 	//layer definitions
// 	vector<function*> S1_functions;
// 	vector<int> S1_rf_width;
// 	vector<int> S1_rf_height;
// 
// 	int S1_input_x_skip = 1;
// 	int S1_input_y_skip = 1;
// 
// 	int S1_output_x_skip = 1;
// 	int S1_output_y_skip = 1;
// 
// 	int S1_width = width / S1_output_x_skip;
// 	int S1_height = height / S1_output_y_skip;
// 
// 	//-------------determine S1 padding and C1 size and skip size
// 	vector<int> S1_padding_width( nr_signalpath );
// 	vector<int> S1_padding_height( nr_signalpath );
// 
// 	vector<int> C1_width( nr_signalpath );
// 	vector<int> C1_height( nr_signalpath );
// 
// 	vector<int> C1_padding_width;
// 	vector<int> C1_padding_height;
// 
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
//  		S1_padding_width.at( i ) = (int) ceil( (double) ( C1_skip_size.at( i ) - 1 ) / 2.0 ) * 2;//(int) ( 2 * ceil( (double) ( ( C1_rf_width.at( i ) - ( S1_width % C1_rf_width.at( i ) ) ) % C1_rf_width.at( i ) ) / 2 ) );
//  		S1_padding_height.at( i ) = (int) ceil( ( (double) C1_skip_size.at( i ) - 1 ) / 2.0 ) * 2;//(int) ( 2 * ceil( (double) ( ( C1_rf_height.at( i ) - ( S1_height % C1_rf_height.at( i ) ) ) % C1_rf_height.at( i ) ) / 2 ) );
// 
// 		C1_width.at( i ) = (int) ceil( (double) S1_width / (double) C1_skip_size.at( i ) );
// 		C1_height.at( i ) = (int) ceil( (double) S1_height / (double) C1_skip_size.at( i ) );
// 
// 		int t_C1_padding_width = s2_receptive_field_size; //TODO is this correct for every rf size?
// 		int t_C1_padding_height = s2_receptive_field_size;
// 
// 		if( t_C1_padding_height % 2 != 0 ) //padding must be even
// 		{
// 			t_C1_padding_height += 1;
// 		}
// 		if( t_C1_padding_width % 2 != 0 )
// 		{
// 			t_C1_padding_width += 1;
// 		}
// 		for( int x = 0;
// 			x < nr_orientation;
// 			x++ )
// 		{
// 			C1_padding_height.push_back( t_C1_padding_height );
// 			C1_padding_width.push_back( t_C1_padding_width );
// 		}
// 		
// 	}
// 
// 	vector<int> s2_width( C1_width.size() );
// 	vector<int> s2_height( C1_height.size() );
// 	copy( C1_width.begin(), C1_width.end(), s2_width.begin() );
// 	copy( C1_height.begin(), C1_height.end(), s2_height.begin() );
// 	int s2_skip = 1;
// //-------------------print layer variables
// 	#ifdef DEBUG
// 	cout << "layer variables..." << endl;
// 	cout << "\tsp\tsize\tpadding\tskip_in\tskip_out\trf" << endl;
// 
// 	cout << "input" << endl;
// 	cout << "\t0\t" << width << "x" << height << "\t" << input_padding_width << "x" << input_padding_height << "\t0x0" << "\t0x0" << "\t0x0" << endl;
// 
// 	cout << "S1" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << S1_width << "x" << S1_height << "\t" <<
// 			S1_padding_width.at( i ) << "x" << S1_padding_height.at( i ) << "\t" <<
// 			S1_input_x_skip << "x" << S1_input_y_skip << "\t" <<
// 			S1_output_x_skip << "x" << S1_output_y_skip << endl;
// 	}
// 
// 	cout << "C1" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << C1_width.at( i ) << "x" << C1_height.at( i ) << "\t" << C1_padding_width.at( i ) << "x" << C1_padding_height.at( i ) << "\t" << C1_skip_size.at( i ) << "x" << C1_skip_size.at( i ) << "\t1x1\t" << C1_receptive_field_size.at( i ) << "x" << C1_receptive_field_size.at( i ) << endl;
// 	}
// 	cout << "S2" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << s2_width.at( i ) << "x" << s2_height.at( i ) << "\t0x0\t" << s2_skip << "x" << s2_skip << "\t" << "1x1\t" << s2_receptive_field_size << "x" << s2_receptive_field_size << endl;
// 	}
// // 	cout << "VTUs for: ";
// // 	copy( object_names.begin(), object_names.end(), ostream_iterator<string>( cout, " " ) );
// // 	cout << endl;
// 	#endif //DEBUG
// //--------------------------------------------------------------------------------------------------
// 	//initialize layers
// 	// --------------layer 0
// 	#ifdef DEBUG
// 	cout << "init input layer" << endl;
// 	#endif //DEBUG
// 	vector<node*> input_layer;
// 	input_layer.push_back(
// 		network.add_node(
// 			FeatureMap<value_type>( width, height, 
// 					input_padding_width, input_padding_height ) ) );
// 
// 	// -----------------S1 layer
// 	#ifdef DEBUG
// 	cout << "adding S1" << endl;
// 	#endif //DEBUG
// 	vector<node*> S1_layer;
// 	int f = 0;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		for( int k = 0; k < nr_orientation; k++ )
// 		{
// 			for( int j = 0; j < filter_bands.at( i ); j++, f++  )
// 			{
// 				stringstream ss( stringstream::in | stringstream::out );
// 				ss << "Gabor" << endl << S1_rf_width.at( f ) << "x" << S1_rf_height.at( f ) << endl << orientation.at( k ) << "°";
// 
// 				S1_layer.push_back( network.add_node( input_layer,
// 					FeatureMap<value_type>( S1_width, S1_height,
// 						S1_padding_width.at( i ), S1_padding_height.at( i ) ),
// 					S1_functions.at( f ),
// 					1,
// 					S1_input_x_skip, S1_input_y_skip,
// 					S1_rf_width.at( f ), S1_rf_height.at( f ),
// 					S1_rf_width.at( f ) - 1, S1_rf_height.at( f ) - 1,
// 					S1_output_x_skip, S1_output_y_skip,
// 					ss.str() ) );
// 			 }
// 		}
// 	}
// 
// 	//---------------------C1 layer
// 	#ifdef DEBUG
// 	cout << "adding C1" << endl;
// 	#endif //DEBUG
// 	vector<node*> C1_layer;
// 
// 	vector<node*>::iterator i_layer = S1_layer.begin();
// 	for( int current_signalpath = 0;
// 		current_signalpath < nr_signalpath;
// 		current_signalpath++ )
// 	{
// 		for( int current_orientation = 0;
// 			current_orientation < nr_orientation;
// 			current_orientation++ )
// 		{
// 			stringstream ss( stringstream::in | stringstream::out );
// 			stringstream info( stringstream::in | stringstream::out );
// 			ss << C1_function;
// 			info << " " << current_signalpath << " " << orientation.at( current_orientation ) << "°";
// 
// 			vector<node*> t_nodes( i_layer, i_layer + filter_bands.at( current_signalpath ) );
// 			i_layer += filter_bands.at( current_signalpath );
// 
// 			C1_layer.push_back( network.add_node( t_nodes,
// 				FeatureMap<value_type>( C1_width.at( current_signalpath ), C1_height.at( current_signalpath ), C1_padding_width.at( current_signalpath ), C1_padding_height.at( current_signalpath ) ),
// 				function_factory::get_function( ss ),
// 				2,
// 				C1_skip_size.at( current_signalpath ), C1_skip_size.at( current_signalpath ),
// 				C1_receptive_field_size.at( current_signalpath ), C1_receptive_field_size.at( current_signalpath ),
// 				S1_padding_width.at( current_signalpath ), S1_padding_height.at( current_signalpath ),
// 				1, 1,
// 				info.str() ) );
// 		}
// 	}
// 	assert( i_layer == S1_layer.end() );
// 
// 	//---------s2
// 	vector<node*> S2_layer = network.add_simple_cell_layer( 2, 3,
// 		s2_composite_features, 
// 		C1_padding_width, C1_padding_height );
// 
// 	//--------------------c2'
// 	#ifdef DEBUG
// 	cout << "adding C2'" << endl;
// 	#endif //DEBUG
// 	vector<node*> C2_prime_layer( S2_layer.size() );
// 	vector<node*>::iterator c = C2_prime_layer.begin();
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		vector<node*>::iterator s2_iterator = S2_layer.begin() + i * nr_s2_features;
// 		for( int j = 0; j < nr_s2_features; j++, s2_iterator++, c++ )
// 		{
// 			vector<node*> pred;
// 
// 			stringstream ss( stringstream::in | stringstream::out );
// 			stringstream info( stringstream::in | stringstream::out );
// 			ss << c2_function;
// 			info << i << " " << j;
// 
// 			pred.push_back( *s2_iterator );
// 
// 			*c = network.add_node( pred,
// 				FeatureMap<value_type>( 1, 1, 0, 0 ),
// 				function_factory::get_function( ss ),
// 				4,
// 				s2_width.at( i ), s2_height.at( i ),
// 				s2_width.at( i ), s2_height.at( i ),
// 				0, 0,
// 				1, 1,
// 				info.str() );
// 		}	
// 	}
// 	//--------------------c2
// 	#ifdef DEBUG
// 	cout << "adding C2" << endl;
// 	#endif //DEBUG
// 	vector<node*> C2_layer( nr_s2_features );
// 	vector<node*>::iterator c2 = C2_layer.begin();
// 	vector<node*>::iterator s2_iterator = C2_prime_layer.begin();
// 	for( int j = 0; j < nr_s2_features; j++, s2_iterator++, c2++ )
// 	{
// 		vector<node*> pred;
// 		for( int i = 0; i < nr_signalpath; i++ )
// 		{
// 			pred.push_back( *( s2_iterator + i * nr_s2_features ) );
// 		}
// 
// 		stringstream ss( stringstream::in | stringstream::out );
// 		stringstream info( stringstream::in | stringstream::out );
// 		ss << c2_function;
// 		info << j;
// 		*c2 = network.add_node( pred,
// 			FeatureMap<value_type>( 1, 1, 0, 0 ),
// 			function_factory::get_function( ss ),
// 			5,
// 			1, 1,
// 			1, 1,
// 			0, 0,
// 			1, 1,
// 			info.str() );
// 	}
// // 
// // // 	#ifdef DEBUG
// // // 	cout << "adding C2''" << endl;
// // // 	#endif //DEBUG
// // // 	vector<node*> VTU_layer;
// // // 	{
// // // 		stringstream ss( stringstream::in | stringstream::out );
// // // 		ss << COMBINE;
// // // 
// // // 		VTU_layer.push_back( network.add_node( C2_layer,
// // // 			FeatureMap<value_type>( 1, nr_combinations, 0, 0 ),
// // // 			function_factory::get_function( ss ),
// // // 			6,
// // // 			1, 1,
// // // 			1, 1,
// // // 			0, 0,
// // // 			1, nr_combinations,
// // // 			"feature vector" ) );
// // // 	}
// // 
// // // 	//-------------------------vtu
// // // 	#ifdef DEBUG
// // // 	cout << "adding VTU" << endl;
// // // 	#endif
// // // 	vector<node*> VTU_layer( nr_objects );
// // // 
// // // 	for( int i = 0; i < nr_objects; i++ )
// // // 	{
// // // 		stringstream ss( stringstream::in | stringstream::out );
// // // 
// // // 		ss << PERCEPTRON << " ";
// // // 		copy( weights.at( i ).begin(), weights.at( i ).end(), ostream_iterator<double>( ss, " " ) );
// // // 
// // // 		VTU_layer.push_back( network.add_node( C2_layer,
// // // 			FeatureMap<value_type>( 1, 1, 0, 0 ),
// // // 			function_factory::get_function( ss ),
// // // 			6,
// // // 			1, 1,
// // // 			1, 1,
// // // 			0, 0,
// // // 			1, 1,
// // // 			object_names.at( i ) ) );
// // // 	}

	return network;
}

// FeatureMapNetwork<Models::T> Models::HMAX_Serre( int width, int height )
// {
// 	//What is happening here:
// 	//	*some global definitions
// 	//	*size, padding, functions, etc are defined for each layer from S1 to c2
// 	//	*most of the variables are printed to std out for debugging
// 	//	*each layer is initialised
// 	int nr_layer = 6;
// 
// 	vector<int> filter_bands;
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 	filter_bands.push_back( 2 );
// 
// 	vector<double> gabor_sigma = _gabor_sigma_serre();
// 	vector<double> gabor_lambda = _gabor_sigma_serre();
// 	double gabor_gamma = 0.3;
// 
// 	assert( gabor_sigma.size() == accumulate( filter_bands.begin(), filter_bands.end(), 0 ) );
// 	assert( gabor_lambda.size() == accumulate( filter_bands.begin(), filter_bands.end(), 0 ) );
// 
// 	int filtersize_inc = 2;
// 
// 	int nr_orientation = 4;
// 	vector<double> orientation( nr_orientation );
// 	double deg = 0;
// 	for( int i = 0; i < nr_orientation; i++ )
// 	{
// 		orientation.at( i ) = deg;
// 		deg += ( 180.0 / nr_orientation );
// 	}
// 
// 	int nr_signalpath = filter_bands.size();
// 	int nr_function_S1 = 0;
// 	for( vector<int>::iterator i = filter_bands.begin();
// 		i != filter_bands.end();
// 		i++ )
// 	{
// 		nr_function_S1 += *i * nr_orientation;
// 	}
// 
// 
// 	vector<int> C1_rf_width;
// 	vector<int> C1_rf_height;
// 	C1_rf_width.push_back( 4 ); C1_rf_height.push_back( 4 );
// 	C1_rf_width.push_back( 5 ); C1_rf_height.push_back( 5 );
// 	C1_rf_width.push_back( 6 ); C1_rf_height.push_back( 6 );
// 	C1_rf_width.push_back( 7 ); C1_rf_height.push_back( 7 );
// 	C1_rf_width.push_back( 8 ); C1_rf_height.push_back( 8 );
// 	C1_rf_width.push_back( 9 ); C1_rf_height.push_back( 9 );
// 	C1_rf_width.push_back( 10 ); C1_rf_height.push_back( 10 );
// 	C1_rf_width.push_back( 11 ); C1_rf_height.push_back( 11 );
// 
// 
// 	double C1_overlay = 2.0;
// 
// 	int s2_receptivefield_size = 2;
// 	int composite_nr_positions = 4;
// 
// 	network network( nr_layer );
// 	network.set_layer_name( 0, "input" );
// 	network.set_layer_name( 1, "S1" );
// 	network.set_layer_name( 2, "C1" );
// 	network.set_layer_name( 3, "S2" );
// 	network.set_layer_name( 4, "C2'" );
// 	network.set_layer_name( 5, "C2" );
// 
// 	cout << "Global variables..." << endl;
// 	cout << "# of signalpaths: " << nr_signalpath << endl;
// 	cout << "# of orientations: " << nr_orientation << endl;
// 	cout << "# of filters per signalpath in S1: ";
// 	copy( filter_bands.begin(), filter_bands.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "# of all filters in S1: " << nr_function_S1 << endl;
// 	cout << "C1 receptive field width:\t";
// 	copy( C1_rf_width.begin(), C1_rf_width.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "C1 receptive field height:\t";
// 	copy( C1_rf_height.begin(), C1_rf_height.end(), ostream_iterator<int>( cout, " " ) ); cout << endl;
// 	cout << "C1 receptive field overlay: " << C1_overlay << endl;
// 	cout << "s2 receptive field size: " << s2_receptivefield_size << " # composite positions: " << composite_nr_positions;
// 
// 	vector<function*> S1_functions;
// 	vector<int> S1_rf_width;
// 	vector<int> S1_rf_height;
// 
// 	int S1_input_x_skip = 1;
// 	int S1_input_y_skip = 1;
// 
// 	int S1_output_x_skip = 1;
// 	int S1_output_y_skip = 1;
// 
// 	int S1_width = width / S1_output_x_skip;
// 	int S1_height = height / S1_output_y_skip;
// 
// 	int input_padding_width = 0;
// 	int input_padding_height = 0;
// 	{
// 		int fs = 7;
// 		vector<double>::iterator i_sigma = gabor_sigma.begin();
// 		vector<double>::iterator i_lambda = gabor_lambda.begin();
// 		for( int i = 0; i < nr_signalpath; i++ )
// 		{
// 			vector<double>::iterator i_sigma_1;
// 			vector<double>::iterator i_lambda_1;
// 			for( int k = 0; k < nr_orientation; k++ )
// 			{
// 				i_sigma_1 = i_sigma;
// 				i_lambda_1 = i_lambda;
// 				int t_fs = fs;
// 				for( int j = 0;
// 					j < filter_bands.at( i );
// 					j++, i_sigma_1++, i_lambda_1++ )
// 				{	
// 					S1_rf_width.push_back( t_fs );
// 					S1_rf_height.push_back( t_fs );
// 			
// 					input_padding_width = max( input_padding_width, t_fs - 1 );
// 					input_padding_height = max( input_padding_height, t_fs - 1 );
// 
// 					vector<vector<double> > filter = gabor( t_fs, t_fs,
// 						*i_sigma_1,
// 						*i_lambda_1,
// 						orientation.at( k ),
// 						gabor_gamma );
// 
// 					stringstream ss( stringstream::in | stringstream::out );
// 
// 					ss << CONVOLUTION << " ";
// 					ss << t_fs << " " << t_fs << " ";
// 					for( vector<vector<double> >::iterator b = filter.begin();
// 						b != filter.end();
// 						b++ )
// 					{
// 						copy( b->begin(), b->end(), ostream_iterator<double>( ss, " " ) );
// 					}
// 					S1_functions.push_back( function_factory::get_function( ss ) );
// 
// 					t_fs += filtersize_inc;
// 				}
// 			}
// 			i_sigma = i_sigma_1;
// 			i_lambda = i_lambda_1;
// 			fs += filtersize_inc * filter_bands.at( i );
// 		}
// 		assert( i_sigma == gabor_sigma.end() );
// 		assert( i_lambda == gabor_lambda.end() );
// 	}
// 
// 	//-------------determine S1 padding and C1 size and skip size
// 	vector<int> S1_padding_width( nr_signalpath );
// 	vector<int> S1_padding_height( nr_signalpath );
// 
// 	vector<int> C1_input_x_skip( nr_signalpath );
// 	vector<int> C1_input_y_skip( nr_signalpath );
// 
// 	vector<int> C1_width( nr_signalpath );
// 	vector<int> C1_height( nr_signalpath );
// 
// 	vector<int> C1_padding_width( nr_signalpath );
// 	vector<int> C1_padding_height( nr_signalpath );
// 
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		C1_input_x_skip.at( i ) = (int) ceil( C1_rf_width.at( i ) / C1_overlay );
// 		C1_input_y_skip.at( i ) = (int) ceil( C1_rf_height.at( i ) / C1_overlay );
// 
//  		S1_padding_width.at( i ) = 2 * (int) ceil( ( ( C1_input_x_skip.at( i ) - ( S1_width % C1_input_x_skip.at( i ) ) ) % C1_input_x_skip.at( i ) ) / 2 );
//  		S1_padding_height.at( i ) = 2 * (int) ceil( ( ( C1_input_x_skip.at( i ) - ( S1_height % C1_input_x_skip.at( i ) ) ) % C1_input_x_skip.at( i ) ) / 2 );
// 
// 		C1_width.at( i ) = (int) ceil( S1_width / (double) C1_input_x_skip.at( i ) );
// 		C1_height.at( i ) = (int) ceil( S1_height / (double) C1_input_y_skip.at( i ) );
// 
// 		C1_padding_width.at( i ) = ( C1_width.at( i ) % s2_receptivefield_size ) * 2; //TODO works fine with s2_rf == 2, not sure about other values
// 		C1_padding_height.at( i ) = ( C1_height.at( i ) % s2_receptivefield_size ) * 2;
// 	}
// 
// 	//-----------------------s2
// 	vector<vector<int> > s2_predecessor = combinations( nr_orientation, composite_nr_positions );
// 	int nr_combinations = s2_predecessor.size();
// 
// 	vector<int> s2_width( nr_signalpath );
// 	vector<int> s2_height( nr_signalpath );
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		s2_width.at( i ) = C1_width.at( i ) / 2 + C1_padding_width.at( i ) / 2; //TODO 
// 		s2_height.at( i ) = C1_height.at( i ) / 2 + C1_padding_width.at( i ) / 2;
// 	}
// 
// //-------------------print layer variables
// 	cout << "layer variables..." << endl;
// 	cout << "\tsp\tsize\tpadding\tskip_in\tskip_out\trf" << endl;
// 
// 	cout << "input" << endl;
// 	cout << "\t0\t" << width << "x" << height << "\t" << input_padding_width << "x" << input_padding_height << "\t0x0" << "\t0x0" << "\t0x0" << endl;
// 
// 	cout << "S1" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << S1_width << "x" << S1_height << "\t" <<
// 			S1_padding_width.at( i ) << "x" << S1_padding_height.at( i ) << "\t" <<
// 			S1_input_x_skip << "x" << S1_input_y_skip << "\t" <<
// 			S1_output_x_skip << "x" << S1_output_y_skip << endl;
// 	}
// 
// 	cout << "C1" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << C1_width.at( i ) << "x" << C1_height.at( i ) << "\t" << C1_padding_width.at( i ) << "x" << C1_padding_height.at( i ) << "\t" << C1_input_x_skip.at( i ) << "x" << C1_input_y_skip.at( i ) << "\t1x1\t" << C1_rf_width.at( i ) << "x" << C1_rf_height.at( i ) << endl;
// 	}
// 	cout << "S2" << endl;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		cout << "\t" << i << "\t" << s2_width.at( i ) << "x" << s2_height.at( i ) << "\t0x0\t" << s2_receptivefield_size << "x" << s2_receptivefield_size << "\t" << "1x1\t" << s2_receptivefield_size << "x" << s2_receptivefield_size << endl;
// 	}
// 	
// //--------------------------------------------------------------------------------------------------
// 	// --------------layer 0
// 	cout << "init input layer" << endl;
// 	vector<node*> layer0;
// 	layer0.push_back(
// 		network.add_node(
// 			FeatureMap<value_type>( width, height, 
// 					input_padding_width, input_padding_height ) ) );
// 
// 	// -----------------S1 layer
// 	cout << "adding S1" << endl;
// 	vector<node*> layer1;
// 	int f = 0;
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		for( int k = 0; k < nr_orientation; k++ )
// 		{
// 			for( int j = 0; j < filter_bands.at( i ); j++, f++  )
// 			{
// 				stringstream ss( stringstream::in | stringstream::out );
// 				ss << "Gabor" << endl << S1_rf_width.at( f ) << "x" << S1_rf_height.at( f ) << endl << orientation.at( k ) << "°";
// 
// 				layer1.push_back( network.add_node( layer0,
// 						network.add_FeatureMap( S1_width, S1_height,
// 							S1_padding_width.at( i ), S1_padding_height.at( i ) ),
// 						S1_functions.at( f ),
// 						1,
// 						S1_input_x_skip, S1_input_y_skip,
// 						S1_rf_width.at( f ), S1_rf_height.at( f ),
// 						S1_rf_width.at( f ) - 1, S1_rf_height.at( f ) - 1,
// 						S1_output_x_skip, S1_output_y_skip,
// 						ss.str() ) );
// 			 }
// 		}
// 	}
// 
// 	//---------------------C1 layer
// 	cout << "adding C1" << endl;
// 	vector<node*> layer2;
// 
// 	vector<node*>::iterator i_layer = layer1.begin();
// 	for( int current_signalpath = 0;
// 		current_signalpath < nr_signalpath;
// 		current_signalpath++ )
// 	{
// 		for( int current_orientation = 0;
// 			current_orientation < nr_orientation;
// 			current_orientation++ )
// 		{
// 			stringstream ss( stringstream::in | stringstream::out );
// 			ss << MAX;
// 
// 			stringstream info( stringstream::in | stringstream::out );
// 			info << current_signalpath << " " << orientation.at( current_orientation ) << "°";
// 
// 			vector<node*> t_nodes( i_layer, i_layer + filter_bands.at( current_signalpath ) );
// 			i_layer += filter_bands.at( current_signalpath );
// 
// 			layer2.push_back( network.add_node( t_nodes,
// 					network.add_FeatureMap( C1_width.at( current_signalpath ), C1_height.at( current_signalpath ), C1_padding_width.at( current_signalpath ), C1_padding_height.at( current_signalpath ) ),
// 					function_factory::get_function( ss ),
// 					2,
// 					C1_input_x_skip.at( current_signalpath ), C1_input_y_skip.at( current_signalpath ),
// 					C1_rf_width.at( current_signalpath ), C1_rf_height.at( current_signalpath ),
// 					S1_padding_width.at( current_signalpath ), S1_padding_height.at( current_signalpath ),
// 					1, 1,
// 					info.str() ) );
// 		}
// 	}
// 	assert( i_layer == layer1.end() );
// 
// 	//---------s2
// 	vector<node*> layer3;
// 
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		vector<node*>::iterator C1_iterator = layer2.begin() + i * nr_orientation;
// 		vector<node*> pred( nr_orientation );
// 		vector<vector<int> >::iterator i_pred = s2_predecessor.begin();
// 		for( int j = 0; j < nr_combinations; j++, i_pred++ )
// 		{
// 			stringstream ss( stringstream::in | stringstream::out );
// 			ss << COMPOSITE_FEATURE;
// 			stringstream info( stringstream::in | stringstream::out );
// 			info << i << "| ";
// 			copy( i_pred->begin(), i_pred->end(), ostream_iterator<int>( info, " " ) );
// 
// 			vector<node*>::iterator i_t = pred.begin();
// 			for( vector<int>::iterator k = i_pred->begin();
// 				k != i_pred->end();
// 				k++, i_t++ )
// 			{
// 				*i_t = *( C1_iterator + *k );
// 			}
// 			assert( i_t == pred.end() );
// 
// 			layer3.push_back( network.add_node( pred,
// 				network.add_FeatureMap( s2_width.at( i ), s2_height.at( i ), 0, 0 ),
// 				function_factory::get_function( ss ),
// 				3,
// 				s2_receptivefield_size, s2_receptivefield_size,
// 				s2_receptivefield_size, s2_receptivefield_size,
// 				C1_padding_width.at( i ), C1_padding_height.at( i ),
// 				1, 1,
// 				info.str() ) );
// 		}
// 	}
// 
// 	//--------------------c2'
// 	cout << "adding S2'" << endl;
// 	vector<node*>layer4( nr_signalpath * nr_combinations );
// 	vector<node*>::iterator c = layer4.begin();
// 	for( int i = 0; i < nr_signalpath; i++ )
// 	{
// 		vector<node*>::iterator s2_iterator = layer3.begin() + i * nr_combinations;
// 		for( int j = 0; j < nr_combinations; j++, s2_iterator++, c++ )
// 		{
// 			vector<node*> pred;
// 
// 			stringstream ss( stringstream::in | stringstream::out );
// 			stringstream info( stringstream::in | stringstream::out );
// 			ss << MAX;
// 			pred.push_back( *s2_iterator );
// 			info << i << " " << j;
// 
// 			*c = network.add_node( pred,
// 				network.add_FeatureMap( 1, 1, 0, 0 ),
// 				function_factory::get_function( ss ),
// 				4,
// 				s2_width.at( i ), s2_height.at( i ),
// 				s2_width.at( i ), s2_height.at( i ),
// 				0, 0,
// 				1, 1,
// 				info.str() );
// 		}
// 		
// 	}
// 	//--------------------c2
// 	cout << "adding S2" << endl;
// 	vector<node*> layer5( nr_combinations );
// 	vector<node*>::iterator c2 = layer5.begin();
// 	vector<node*>::iterator s2_iterator = layer4.begin();
// 	for( int j = 0; j < nr_combinations; j++, s2_iterator++, c2++ )
// 	{
// 		vector<node*> pred;
// 		for( int i = 0; i < nr_signalpath; i++ )
// 		{
// 			pred.push_back( *( s2_iterator + i * nr_combinations ) );
// 		}
// 
// 		stringstream ss( stringstream::in | stringstream::out );
// 		stringstream info( stringstream::in | stringstream::out );
// 		ss << MAX;
// 		info << j;
// 
// 		*c2 = network.add_node( pred,
// 			network.add_FeatureMap( 1, 1, 0, 0 ),
// 			function_factory::get_function( ss ),
// 			5,
// 			1, 1,
// 			1, 1,
// 			0, 0,
// 			1, 1,
// 			info.str() );
// 	}
// 
// 	assert( c == layer4.end() );
// 	return network;
// }

// vector<FeatureMapNode<Models::value_type> > Models::_add_signalpath( network& network,
// 	int signal_path, //TODO
// 	int layer,
// 	vector<vector<node*> >& predecessors,
// 	vector<function*> functions,
// 	int output_padding_width, int output_padding_height,
// 	int input_skip_width, int input_skip_height,
// 	int output_skip_width, int output_skip_height,
// 	vector<int>& receptive_field_width, vector<int>& receptive_field_height,
// 	const string& description )
// {
// 	vector<node*> r;
// 	for( int i = 0; i < functions.size(); i++ )
// 	{
// 		#ifdef DEBUG
// 		{
// 			//Assume that all predecessor activations have same size
// 			int width = predecessors.at( i ).front().activation().width();
// 			int height = predecessors.at( i ).front().activation().height();
// 			for( vector<node*>::iterator j = predecessors.at( i ).begin();
// 				j != predecessors.at( i ).end();
// 				j++ )
// 			{
// 				assert( j->activation().width() == width );
// 				assert( j->activation().height() == height );
// 	
// 			}
// 		}
// 		#endif //DEBUG
// 		int width = 0; //TODO
// 		int height = 0;
// 		int input_padding_width = 2 * (int) ceil(
// 			( ( input_skip_width -
// 				( predecessors.at( i ).front().activation().width() % input_skip_width ) )
// 				% input_skip_width )
// 			/ 2 );
// 		int input_padding_height = 2 * (int) ceil(
// 			( ( input_skip_height -
// 				( predecessors.at( i ).front().activation().height() % input_skip_height ) )
// 				% input_skip_height )
// 			/ 2 ); //TODO debug assert( input_padding_width > predecessor_FeatureMap_width )...
// 
// 		r.push_back( network.add_node( predecessors.at( i ),
// 				FeatureMap<value_type>( width, height, output_padding_width, output_padding_height ),
// 				functions.at( i ),
// 				layer,
// 				input_skip_width, input_skip_height,
// 				receptive_field_width.at( i ), receptive_field_height.at( i ),
// 				output_skip_width, output_skip_height,
// 				input_padding_width, input_padding_height,
// 				description ) );
// 	}
// 
// 	return r;
// }

 pair<vector<string>, vector<Models::function*> > Models::functions_from_strings( const vector<string>& function_strings )
{
	vector<function*> r;
	vector<string> d;

	for( vector<string>::const_iterator i = function_strings.begin();
		i != function_strings.end();
		i++ )
	{
		stringstream ss( stringstream::in | stringstream::out );
		ss.str( (*i) );
		
		string s;

		ss >> s;
		
		d.push_back( s );
		r.push_back( function_factory::get_function( ss ) );
	}
	return pair<vector<string>, vector<function*> >( d, r );
}

pair<int, int> Models::padding_size( const vector<Models::function*>& functions )
{
	int width = 0;
	int height = 0;

	for( vector<function*>::const_iterator i = functions.begin();
		i != functions.end();
		i++ )
	{
		function* f = *i;

		if( f->width() > width )
		{
			width = f->width();
		}
		if( f->height() > height )
		{
			height = f->height();
		}
	}
// 	cout << width << " " << height << endl;
// 	width = width / 2;
// 	height = height / 2;
	if( width % 2 != 0 )
	{
		width++;
	}
	if( height % 2 != 0 )
	{
		height++;
	}

	return pair<int, int>( width * 2, height * 2 );
}
