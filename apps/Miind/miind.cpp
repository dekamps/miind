#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <PopulistLib/PopulistLib.h>
#include <MiindLib/SimulationCode.h>
#include <MiindLib/SimulationBuilderCode.h>
#include <MiindLib/SimulationParserCode.h>

using PopulistLib::PopulationConnection;
using MiindLib::SimulationParser;
using std::cout;
using std::endl;
using std::string;

namespace po = boost::program_options;

int main(int argc, char** argv)
{
	try{
		bool b_batch = false;
		bool b_fig   = false;
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("input-file",	po::value< vector<string> >(),	"input file")
			("g",			po::value<string>(),			"generate a simulation file")
			("w",			po::value<string>(),			"weight type, default is 'double' (only with --g)")
			("fig",			po::bool_switch(&b_fig),		"svg files produced after simulation")
			("b",			po::bool_switch(&b_batch),		"batch mode");
			;
		if (b_batch)
			cout << "Forcing batch mode" << endl;

		po::positional_options_description p;
		p.add("input-file", -1);
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).
        options(desc).positional(p).run(), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << desc << "\n";
			return 1;
		}

		if (vm.count("input-file") > 1 )
		{
			cout << "Please specify one input file" << endl;
			return 0;
		}
		if (vm.count("input-file") == 1)
		{
			cout << "Input file is: " << vm["input-file"].as< vector<string> >()[0] << "\n";
			if (b_fig){
				ofstream ofst("test/test.txt");
				if (! ofst){
					cout << "Figure production requested. Make a directory called 'test' and run again" << endl;
					exit(0);
				}
			}
			// simply build simulation from file
			SimulationParser parse;
			parse.ExecuteSimulation(vm["input-file"].as< vector<string> >()[0], b_batch );
			if (b_fig)
				parse.Analyze(vm["input-file"].as< vector<string> >()[0]);
		}
		else {
				if (! vm.count("g") )
				{
					cout << "You have specified no input file" << endl;
					return 0;
				}
				else {
	
					if ( ! vm.count("w") ) 
					{
						SimulationParser parse;
						parse.GenerateXMLFile(vm["g"].as<string>(),"double");
					}
					else
					{
						SimulationParser parse;
						parse.GenerateXMLFile(vm["g"].as<string>(),vm["w"].as<string>());
					}
				}
		}
	}
	catch(UtilLib::GeneralException& exc)
	{
		cerr << "MIIND execption: " << exc.Description() << endl;
		return 1;
	}
	catch(exception& e) {
		cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch(...)
	{
		cerr << "Unknown exception" << endl;
		return 1;
	}
    return 0;
}
