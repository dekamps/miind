#include <gtkmm/main.h>

#include <Magick++.h>

#include <LayerMappingLib/LayerMappingLib.h>

#include <iostream>
#include <fstream>

using namespace std;

using namespace Magick;

using namespace LayerMappingLib;
using namespace LayerMappingLib::gtkmm;

void usage( const string& );
vector<double> feature_vector( char* filename );
NetworkInterface<network_ensemble> hmax_serre( int width, int height, string s1_features_file  );
vector<string> read_filter( string& file_name );

#include <complex>
#include <fftw3.h>

int main (int argc, char *argv[])
{
	if( argc < 4 )
	{
		usage( argv[ 0 ] );
		return -1;
	}
	Gtk::Main kit( argc, argv );

	string image_name = argv[ 1 ];
	vector<double> v = feature_vector( argv[ 2 ] );
	string s1_features = argv[ 3 ];

	LayerMappingApplet layer_mapping_applet;
	layer_mapping_applet.set_feedback_vector( v );

	Image input_image;
	try
	{
		input_image.read( image_name );
	}
	catch( LayerMappingLib::Exception &error_ )
	{
		cout << "Caught exception: " << error_.what() << endl;
		return 1;
	}

	int width = input_image.size().width();
	int height = input_image.size().height();

	//initialize weights for vtus
	vector<vector<value_type> > weights; //TODO read from file
	vector<string> object_names;

// 	weights.push_back( vector<value_type>( 256 ) );//TODO read from file
// 	weights.push_back( vector<value_type>( 256 ) );
//
// 	object_names.push_back( "circle" );
// 	object_names.push_back( "rectangle" );

// 	NetworkInterface<network_ensemble> ne = Models::HMAX(
// 		width,
// 		height,
// 		"Max", "Max"  );

//	NetworkInterface<network_ensemble> ne = hmax_serre( width, height, s1_features );

 	NetworkInterface<network_ensemble> ne = Models::HMAX_Feedback(
 		width,
 		height,
 		"Max", "Max"  );


// 	network n = Models::SimpleTest( width, height );
// 
// 	network_ensemble ne;
// 	ne.add_network( n );

// 	NetworkInterface<network_ensemble> ne = Models::SimpleEnsemble(
// 		width,
// 		height );

// 	NetworkInterface<network_ensemble> ne = Models::SimpleFeedback(
// 		width,
// 		height );

// 	NetworkInterface<network_ensemble> ne = Models::ConvolutionTest(
// 		width,
// 		height,
// 		21, 21 );

	//set input activation to image
	double t[ width * height ];

	input_image.quantizeColorSpace( GRAYColorspace );
	input_image.write( 0, 0, width, height, "R", DoublePixel, t );

	ne.set_input( t );

	layer_mapping_applet.setNetworkEnsemble( ne.network() );
	layer_mapping_applet.show();

	Gtk::Main::run( layer_mapping_applet );

	return 0;
}

void usage( const string& name )
{
	cout << "Usage: " << name << " <image_file_name> <feedback_vector>" << endl << endl;
	cout << "	Where the image has to be in a format supported by ImageMagick." << endl;
	cout << "	Where feedback_vector is the feature vector of the object to attent to. The file format is acii, with spaces separated." << endl;
}

vector<string> read_filter( string& file_name )
{
	vector<string> r;

	fstream file;
	file.open( file_name.c_str(), fstream::in );

 	while( file.good() )
	{
		string filter;

		getline( file, filter );

		r.push_back( filter );	
	}
	r.pop_back();

	return r;
}

NetworkInterface<network_ensemble> hmax_serre( int width, int height, string s1_features_file )
{
	vector<string> s1_features = read_filter( s1_features_file );
	cout << "filter matrices in s1 " << s1_features.size() << endl;

	vector<int> c1_receptive_field_size;
	c1_receptive_field_size.push_back( 8 );
	c1_receptive_field_size.push_back( 10 );
	c1_receptive_field_size.push_back( 12 );
	c1_receptive_field_size.push_back( 14 );
	c1_receptive_field_size.push_back( 16 );
	c1_receptive_field_size.push_back( 18 );
	c1_receptive_field_size.push_back( 20 );
	c1_receptive_field_size.push_back( 22 );

	vector<int> c1_skip_size;
	c1_skip_size.push_back( 3 );
	c1_skip_size.push_back( 5 );
	c1_skip_size.push_back( 7 );
	c1_skip_size.push_back( 8 );
	c1_skip_size.push_back( 10 );
	c1_skip_size.push_back( 12 );
	c1_skip_size.push_back( 13 );
	c1_skip_size.push_back( 15 );

	vector<int> filter_bands;
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	filter_bands.push_back( 2 );
	vector<string> s2_features;
	s2_features.push_back( "bla Convolution 3 3 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000" );
	s2_features.push_back( "bla Convolution 3 3 1.000000 1.000000 0.000000 0.000000 1.000000 1.000000 0.000000 1.000000 0.000000" );
	s2_features.push_back( "bla Convolution 3 3 0.000000 1.000000 0.000000 1.000000 1.000000 1.000000 0.000000 0.000000 0.000000" );
	
	NetworkInterface<network_ensemble> ne = Models::HMAX_Learned_S2(
		width, height,
		filter_bands,
		s1_features,
		"Max", c1_receptive_field_size, c1_skip_size,
		s2_features,
		"Max"  );

	ne.fill_feature_map_padding_with_noise( 0.0 );

	return ne;
}

vector<double> feature_vector( char* filename )
{
	cout << "reading from " << filename << endl;

	ifstream file;
	file.open( filename );

	vector<double> r;
	while( !file.eof() )
	{
		double x;
		file >> x;
// 		cout << x << endl;
		
		r.push_back( x );
	}

	file.close();

	cout << "feature vector loaded from " << filename << " length " << r.size() << endl;
	return r;
}
