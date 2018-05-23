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

	if (argc != 5 && argc != 7 && argc != 8 && argc != 9 && argc != 10)
		throw TwoDLib::TwoDLibException("Incorrect number of arguments. Usage is either: ./MatrixGenerator <basename>.model <basename>.fid n_points tr_v tr_w tr_reset [n_min] [n_max] [-use_area_calculation], or ./MatrixGenerator <basename>.model <basename>.fid n_points <basename>.jmp [n_min] [n_max].");
	// if argv[4] is the jump file, then argc must be 5 or 7

	std::string translation(argv[4]);
	std::vector<string> elem;
	TwoDLib::split(translation,'.',elem);
	if (elem.size() < 2 || elem[1] != string("jmp")){
		// then it must be a number
		// still here?, then argc should be 7 or 9 and the mode is TranslationArguments
		if (argc == 7 || argc == 9)
			return TwoDLib::TranslationArguments;
		if (argc == 8 || argc == 10)
			return TwoDLib::AreaCalculation;
		else
			throw TwoDLib::TwoDLibException("You should have 7 or 9 arguments without a jump file.");
	} else {
		// ok, so this is a jump file. the number of arguments should be 5 or 7
		if (argc == 5 || argc == 7)
			return TwoDLib::JumpFile;
		else
			throw TwoDLib::TwoDLibException("You should have 5 or 7 arguments with a jump file.");
	}
}

int main(int argc, char** argv){

	try {
		TwoDLib::UserTranslationMode mode = InterpretArguments(argc, argv);
		TwoDLib::GenerateMatrix(argc,argv,mode);
	}
	catch(const TwoDLib::TwoDLibException& excep){
		std::cout << excep.what() << std::endl;
	}

	return 0;
}
