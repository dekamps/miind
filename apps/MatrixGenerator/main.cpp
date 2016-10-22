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
#include <TwoDLib.hpp>
#include "Bind.hpp"
#include "CorrectStrays.hpp"
#include "GenerateMatrix.hpp"
#include "ConstructResetMapping.hpp"

using namespace std;


enum UseCase { Binding, MatrixGeneration };

UseCase InterpretArguments(int argc, char** argv){
	// better test immediately so that existence of argv[1] is guaranteed.
	if (argc != 6 && argc != 7  && argc != 9)
		throw TwoDLib::TwoDLibException("Incorrect number of arguments. Usage is either: ./MatrixGenerator <basename>.mesh <basename>.stat <basename>.rev V_res theta, or ./MatrixGenerator <basename>.model <basename>.fid n_points tr_v tr_w tr_reset [n_min] [n_max].");
	std::string mesh_name(argv[1]);
	std::vector<string> elem;
	TwoDLib::split(mesh_name,'.',elem);

	if (argc == 6 && elem[1] == string("mesh"))
		return Binding;
	if (argc == 7 || argc == 9)
		return MatrixGeneration;
	// should never get here, but compiler doesn't know
	throw TwoDLib::TwoDLibException("Interpret argument failed in MatrixGenerator.");
}

int main(int argc, char** argv){

	try {

		if (InterpretArguments(argc,argv) == Binding)
			TwoDLib::Bind(argc,argv);
		else if (InterpretArguments(argc, argv) == MatrixGeneration)
			TwoDLib::GenerateMatrix(argc,argv);
		else
			throw TwoDLib::TwoDLibException("Don't understand use case.");
	}
	catch(const TwoDLib::TwoDLibException& excep){
		std::cout << excep.Description() << std::endl;
	}

	return 0;
}
