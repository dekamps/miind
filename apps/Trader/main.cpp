// DynamicLibTest.cpp : Defines the entry point for the console application.
//


#include <sstream>
#include "NetworkedModelSimulator.h"

using std::ostringstream;

int main(int argc, char** argv)
{

	double b = 2.5;
	double c = 1.0;	
	string nettype("striped_torus");	

	if (argc == 4){
		cout << "The desired nettype is: " << nettype << endl;
		ostringstream ost;
		nettype = argv[1];
		b = atof(argv[2]);
		c=  atof(argv[3]);
	}

	NetworkedModelSimulator networks;

/*
	double c_min = 1.0;  // production value 0.0
	double c_max = 2.0;  // production value 2.0
	double b_min = 2.5;  // production value 0.0
	double b_max = 5.0;  // production value 5.0

	int n_b = 1; // production value 20
	int n_c = 1; // production value 20

	// The logic being tha if only one point is requested, the loop below should only run once
	double delta_b = (n_b < 2) ? 0 : (b_max - b_min)/(n_b - 1);
	double delta_c = (n_c < 2) ? 0 : (c_max - c_min)/(n_c - 1);

	cout << delta_b << endl;
*/


	networked_simulation_parameter param(nettype);

	
	param.b = b; 
	param.c = c;

	for (int i_version = 0; i_version < 100; i_version++)
		for (int simulation_type = 0; simulation_type < 3; simulation_type++)
		{
			ostringstream str;
			str << nettype <<"_b_" << param.b << "_c_" << param.c << "_i_version_" << i_version << "_type_" << simulation_type << ".root";
			cout << str.str() << endl;
			param.out_filename = str.str();
			param._type = static_cast<networked_simulation_parameter::SimulationType>(simulation_type);
			networks.RunStar(param);
		}
	


	return 0;
}

