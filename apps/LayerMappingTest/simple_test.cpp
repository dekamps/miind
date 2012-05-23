/* This is maybe the simplest test program for the LayerMappingLib
 * and used in our paper covering miind.
 * added by Volker Baier Okt.24th 2007
 */

#include <iostream>
#include <iterator>
#include <vector>
#include <iomanip>
#include <fstream>

#include <LayerMappingLib/LayerMappingLib.h>

using namespace std;
using namespace LayerMappingLib;

int main( int argc, char** argv )
{
  Models::network myModel = Models::SimpleTest( 16, 16 );

  FeatureMap<double> input_layer = myModel.layer_activation( 0 ).front();


  generate( input_layer.begin(), input_layer.end(), rand );

  evolve( myModel.begin(), myModel.end() );

#ifdef DEBUG
  myModel.debug_print();
#endif
}
