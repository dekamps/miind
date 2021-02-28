/*
 * main.cpp
 *
 *  Created on: Jan 21, 2016
 *      Author: scsmdk
 */
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include "Bind.hpp" // for the definitions of split
#include <TwoDLib.hpp>
#include "CorrectStrays.hpp"
#include "GenerateMatrix.hpp"
#include "ConstructResetMapping.hpp"

using namespace std;


TwoDLib::UserTranslationMode InterpretArguments(int argc, char** argv){

	if (argc < 5)
		throw TwoDLib::TwoDLibException("Incorrect number of arguments. Usage is either: ./MatrixGenerator <mode> <basename>.model <basename>.fid n_points tr_v tr_w tr_reset [n_min] [n_max] [-use_area_calculation], or ./MatrixGenerator <mode> <basename>.model <basename>.fid n_points <basename>.jmp [n_min] [n_max].\n\nWhere mode can be:\nmc : Monte Carlo\narea : Area Calculation\njump : Use a jump file (with Monte Carlo)\nreset : Recalculate reset cells only.\ntransform : Calculate the dynamics transition matrix for the Grid method.\nresettransform : Recalculate reset cells only for Grid method.\n\n");
	// if argv[4] is the jump file, then argc must be 5 or 7

	std::string mode(argv[1]);
	if (mode == string("jump"))
		return TwoDLib::JumpFile;
	if (mode == string("area"))
		return TwoDLib::AreaCalculation;
	if (mode == string("mc"))
		return TwoDLib::TranslationArguments;
	if (mode == string("reset"))
		return TwoDLib::ResetOnly;
	if (mode == string("transform"))
		return TwoDLib::Transform;
	if (mode == string("resettransform"))
		return TwoDLib::TransformResetOnly;

}

int main(int argc, char** argv){

	std::cout << "Running MatrixGenerator.\n" << std::flush;

	try {
		TwoDLib::UserTranslationMode mode = InterpretArguments(argc, argv);
		TwoDLib::GenerateMatrix(argc,argv,mode);
	}
	catch(const TwoDLib::TwoDLibException& excep){
		std::cout << excep.what() << std::endl;
	}

	return 0;
}
