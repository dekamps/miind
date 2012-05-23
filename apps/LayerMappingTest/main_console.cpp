#include <iostream>
#include <iterator>
#include <vector>
#include <iomanip>
#include <fstream>

#include <Magick++.h>

#include <LayerMappingLib/LayerMappingLib.h>

using namespace std;

using namespace Magick;

using namespace LayerMappingLib;


void usage( const std::string& );
void write_featuremaps( NetworkInterface<network_ensemble> ne, const string& featuremap_directory );
void write_hmax_featurevector( NetworkInterface<network_ensemble> ni, const string& feature_vector_name );

int main( int argc, char** argv )
{
	if( argc < 3 )
	{
		usage( argv[ 0 ] );
		return -1;
	}
	string image_name = argv[ 1 ];
	string feature_vector_name = argv[ 2 ];
	string featuremap_directory;
	if( argc == 4 )
	{
		featuremap_directory = argv[ 3 ];
	}

	//load image from file
	Image input_image;
	try
	{
		input_image.read( image_name );
	}
	catch( LayerMappingLib::Exception& error_ )
	{
		cout << "Caught exception: " << error_.what() << endl;
		return 1;
	}
	int width = input_image.size().width();
	int height = input_image.size().height();

// 	//initialize weights for vtus
// 	vector<vector<value_type> > weights; //TODO read from file
// 	vector<std::string> object_names;

	//build network
	NetworkInterface<network_ensemble> ne = Models::HMAX(
		width,
		height,
		"Max", "Max" );

// 	network_ensemble ne = Models::ConvolutionTest( width, height, 13, 13 );

	//set input activation to image
	double t[ width * height ];

	input_image.quantizeColorSpace( GRAYColorspace );
	input_image.write( 0, 0, width, height, "R", DoublePixel, t );

	ne.set_input( t );

// 	evolve
	cout << "Evolving..." << endl;
	ne.evolve();
	cout << "done" << endl;

	//write feature vector
	cout << "writing feature vector" << endl;
	write_hmax_featurevector( ne, feature_vector_name );
	cout << "done" << endl;

// 	write images to files
	if( !featuremap_directory.empty() )
	{
		cout << "writing images..." << endl;
		write_featuremaps( ne, featuremap_directory );
		cout << endl << "done" << endl;
	}
	return 0;
}

void write_featuremaps( NetworkInterface<network_ensemble> ne, const string& featuremap_directory )
{
		network_ensemble n = ne.network();

		int nr = 0;
		for( network_ensemble::iterator i = n.begin();
			i != n.end();
			i ++ )
		{
			for( network_ensemble::network::iterator node = i->begin();
				node != i->end();
				node++, nr++ )
			{
				int width = node->activation().width();
				int height = node->activation().height();

				stringstream ss( stringstream::in | stringstream::out );
				ss << featuremap_directory << "/" << setfill( '0' ) << setw( 4 ) << nr << node->description() << ".png";

				double data[ width * height * 3 ];
				double* d = data;
				for( network::node::iterator i = node->activation().begin();
					i != node->activation().end();
					i++, d++ )
				{
					*d = *i; d++;
					*d = *i; d++;
					*d = *i;
				}
				Image image;

				image.read( width, height, "RGB", DoublePixel, data );
				image.write( ss.str() );
			}
		}
}

void write_hmax_featurevector( NetworkInterface<network_ensemble> ni, const string& feature_vector_name )
{
	network_ensemble ne = ni.network();

	vector<double> feature_vector;
	for( network::iterator i = ne.begin()->begin( ne.begin()->nr_layers() - 2 );
		i != ne.begin()->end( ne.begin()->nr_layers() - 2 );
		i++ )
	{
		feature_vector.push_back( *(i)->activation().begin() );
	}
	ofstream f( ( feature_vector_name + ".mat" ).c_str() );
	f << "# name: " << feature_vector_name << endl
		<< "# type: matrix" << endl
		<< "# rows: " << feature_vector.size() << endl
		<< "# columns: " << "1" << endl;
	copy( feature_vector.begin(),
		feature_vector.end(),
		ostream_iterator<double>( f, "\n" ) );
}

void usage( const std::string& name )
{
	cout << "Usage: " << name << " <input_image> <feature_vector> [featuremap_directory]" << endl << endl;
	cout << "	Where the image has to be a format supported by ImageMagick." << endl;
	cout << "	The feature vector will be stored in 'feature_vector'.mat." << endl;
	cout << "	The feature maps are stored in the directory specified by featuremap_directory." << endl;
}
