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


enum UseCase { Binding, MatrixGeneration, MatrixGeneration };

UseCase InterpretArguments(int argc, char** argv){
	// better test immediately so that existence of argv[1] is guaranteed.
	if (argc != 4 && argc != 6 && argc != 7 && argc != 8 && argc != 9 && argc != 10)
		throw TwoDLib::TwoDLibException("Incorrect number of arguments. Usage is: ./MatrixGenerator mode <basename>.model <basename>.fid n_points tr_v tr_w tr_reset [n_min] [n_max].\n\nWhere mode can be:\nmc : Monte Carlo\narea : Area Calculation\njump : Use a jump file (with Monte Carlo)\nreset : Recalculate reset cells only.\ntransform : Calculate the dynamics transition matrix for the Grid Method.\nresettransform : Recalculate reset cells only for Grid method.\n\n");
	std::string mesh_name(argv[1]);
	std::vector<string> elem;
	TwoDLib::split(mesh_name,'.',elem);

	if (argc == 4 && elem[1] == string("mesh"))
		return MatrixGeneration;

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
		std::cout << excep.what() << std::endl;
	}

	return 0;
}
