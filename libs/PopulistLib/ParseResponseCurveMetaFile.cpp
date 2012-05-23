// Copyright (c) 2005 - 2010 Marc de Kamps
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
#include <fstream>
#include "../UtilLib/UtilLib.h"
#include "ParseResponseCurveMetaFile.h"
#include "FitRateComputation.h"
#include "IntegralRateComputation.h"
#include "InterpolationRebinner.h"

using namespace PopulistLib;
using namespace UtilLib;
using namespace std;

	const double    RESPONSE_CURVE_EXPANSION_FACTOR	= 1.1;		// maximum expansion before rebinning
	const Number    RESPONSE_CURVE_NADD				= 1;		// add one bin at a time
	const Number	RESPONSE_CURVE_MAX_ITER			= 1000000;	// maximum number of iterations allowed

	const Number    RESPONSE_CURVE_SINGLE_NBINS		= 2200;  // number of bins at start of simulation
	const Number	RESPONSE_CURVE_DOUBLE_NBINS		= 550;
	const Density   RESPONSE_CURVE_D_MAX            = 100;


	const InitialDensityParameter
		RESPONSE_CURVE_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

	const FitRateComputation		FIT_RATE_COMPUTATION;
	const IntegralRateComputation	INTEGRAL_RATE_COMPUTATION;
	const InterpolationRebinner		REBIN_INTERPOL;


ParseResponseCurveMetaFile::ParseResponseCurveMetaFile(const std::string & path_name):
_path_name(path_name){
	if (! Parse())
		WriteDefaultFile();
}

PopulationParameter ParseResponseCurveMetaFile::ParPop() const
{
	return _par_pop;
}

SimulationRunParameter ParseResponseCurveMetaFile::ParSim
(
	const AbstractReportHandler& handler,
	const string& log
) const
{
	return 
		SimulationRunParameter
		(
			handler,
			RESPONSE_CURVE_MAX_ITER,
			_t_start,
			_t_end,
			_t_report,
			_t_update,
			_t_net,
			log,
			_t_report
		);
}

void ParseResponseCurveMetaFile::WriteDefaultFile() const {
	ofstream str(_path_name.c_str());

	str << "<LIFNeuronParameter>\n"; 
	str << "<tau_mem_in_s>"			<< "\t" <<	"20e-3"		<< "\t" <<	"</tau_mem_in_s>\n";
	str << "<E_L_in_V>"				<< "\t"	<<	"0.0"		<< "\t" <<	"</E_L_in_V>\n";
	str << "<V_reset_in_V>"			<< "\t" <<	"10.0e-3"	<< "\t" <<	"</V_reset_in_V>\n";
	str << "<V_threshold_in_V>"		<< "\t" <<	"20e-3"		<< "\t" <<	"</V_threshold_in_V>\n";
	str << "<tau_ref_in_s>"			<< "\t" <<	"0.0"		<< "\t" <<	"</tau_ref_in_s>\n";
	str << "</LIFNeuronParameter>\n";
	str << "<SimulationRunParameter>\n";
	str << "<t_start_in_s>"			<< "\t" <<	"0.0"		<< "\t" <<	"</t_start_in_s>\n";
	str << "<t_end_in_s>"			<< "\t" <<  "1.0"		<< "\t" <<	"</t_end_in_s>\n";
	str	<< "<t_networkstep_in_s>"	<< "\t" <<	"1e-3"		<< "\t" <<	"</t_networkstep_in_s>\n";
	str	<< "<t_report_in_s>"		<< "\t"	<<	"1e-2"		<< "\t"	<<	"</t_report_in_s>\n";
	str	<< "<t_update_in_s>"		<< "\t" <<	"1e-2"		<< "\t" <<	"</t_update_in_s>\n";
	str << "</SimulationRunParameter>\n";
	str << "<ParameterScan>\n";
	str << "<Sequence><Incremental><mu_in_mV 10e-3 21e-3 1e-3></Incremental></Sequence>\n";
	str << "<Sequence><Series><sigma_in_mV 1e-3 2e-3 5e-3 7e-3></Series></Sequence>\n";
	str << "</ParameterScan>\n";
}

bool ParseResponseCurveMetaFile::Parse(){
	_p_str_current = new ifstream(_path_name.c_str()); 
	if (! _p_str_current ) 
		return false;

	if (! this->ParsePopulationParameter(*_p_str_current) )
		return false;

	if (! this->ParseSimulationParameter(*_p_str_current) )
		return false;

	_p_scan =  boost::shared_ptr<ParameterScan>(new ParameterScan(*_p_str_current));

	return true;
}

bool ParseResponseCurveMetaFile::ParsePopulationParameter(istream& str)
{
	string str_in;
	str >> str_in;
	if (str_in != string("<LIFNeuronParameter>") )
		return false;

	str >> str_in;
	if (str_in != string("<tau_mem_in_s>"))
		return false;
	str >> _par_pop._tau;
	str >> str_in;
	if (str_in != string("</tau_mem_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("<E_L_in_V>"))
		return false;
	str >> _par_pop._V_reversal;
	str >> str_in;
	if (str_in != string("</E_L_in_V>"))
		return false;

	str >> str_in;
	if (str_in != string("<V_reset_in_V>"))
		return false;
	str >> _par_pop._V_reset;
	str >> str_in;
	if (str_in != string("</V_reset_in_V>"))
		return false;

	str >> str_in;
	if (str_in != string("<V_threshold_in_V>"))
		return false;
	str >> _par_pop._theta;
	str >> str_in;
	if (str_in != string("</V_threshold_in_V>"))
		return false;

	str >> str_in;
	if (str_in != string("<tau_ref_in_s>"))
		return false;
	str >> _par_pop._tau_refractive;
	str >> str_in;
	if (str_in != string("</tau_ref_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("</LIFNeuronParameter>") )
		return false;

	return true;
}

bool ParseResponseCurveMetaFile::ParseSimulationParameter(istream& str){

	string str_in;
	str >> str_in;
	if (str_in != string("<SimulationRunParameter>") )
		return false;

	str >> str_in;
	if (str_in != string("<t_start_in_s>"))
		return false;
	str >> _t_start;
	str >> str_in;
	if (str_in != string("</t_start_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("<t_end_in_s>"))
		return false;
	str >> _t_end;
	str >> str_in;
	if (str_in != string("</t_end_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("<t_networkstep_in_s>"))
		return false;
	str >> _t_net;
	str >> str_in;
	if (str_in != string("</t_networkstep_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("<t_report_in_s>"))
		return false;
	str >> _t_report;
	str >> str_in;
	if (str_in != string("</t_report_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("<t_update_in_s>"))
		return false;
	str >> _t_update;
	str >> str_in;
	if (str_in != string("</t_update_in_s>"))
		return false;

	str >> str_in;
	if (str_in != string("</SimulationRunParameter>") )
		return false;
	//catch the newline
    getline(str,str_in);
	return true;
}

ParameterScan& ParseResponseCurveMetaFile::ParScan() const
{
	return *_p_scan;
}

PopulistSpecificParameter ParseResponseCurveMetaFile::ParSpec() const
{
	Potential V_min			= -0.1*_par_pop._theta;
	return 
		PopulistSpecificParameter
		( 
			V_min,
			RESPONSE_CURVE_SINGLE_NBINS,
			1,
			RESPONSE_CURVE_INITIAL_DENSITY,
			RESPONSE_CURVE_EXPANSION_FACTOR
		);
	
}