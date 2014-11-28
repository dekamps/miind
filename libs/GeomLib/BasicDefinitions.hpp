// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_GEOMLIB_BASICDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_BASICDEFINITIONS_INCLUDE_GUARD

#include "../UtilLib/UtilLib.h"

using UtilLib::Number;

//! GeomLib

namespace GeomLib
{
	//! typedef floating point number between or equal to 0 and 1
	typedef double Fraction;

/*	//! define Efficacy as type
	typedef double Efficacy;

	//! define Time as a type
	typedef double Time;

	//! define Rate as a type
	typedef double Rate;

	//! define membrane potential as type
	typedef double Potential;

	//! define Density as type
	typedef double Density;

	//! Define Probability as type
	typedef double Probability;

	//! Define Capacity as type
	typedef double Capacity;

	//! Define Conductance as type
	typedef double Conductance;

	//! Define Current as type
	typedef double Current;

	//! Define anything that has a state as a type
	typedef double State;

	//! a large value, but not numeric_limits::max, since this value is used in divisions
	//! this corresponds to a membrane time constant of twenty minuts
	const Time TIME_MEMBRANE_INFINITE = 1000000; 
*/
	/*	const Number NUMBER_POPULIST_PARAMETERS = 4;
	
	const int GSL_ERROR_HANDLER_ACTIVE = 0;

	//! if a stepsize is calculated to larger than DIFFUSION_LIMIT\f$ (\theta -V_{rev})\f$, a single input cannot mimmck a diffusion process any more
	const double DIFFUSION_LIMIT = 0.05; 	
	
	//! if a double input must mimmick a diffusion process, then a small efficacy must be chosen
	const double DIFFUSION_STEP = 0.03;

	//! Membrane Capacity in Brette & Gerstner (2005)
	const Capacity BG_C_M    = 281;    //pf; membrane capacity

	//! Leak Conductance in Brette & Gerstner (2005)
	const Conductance BG_G_L = 30;     //nS; leak conductance

	//! Leak Potential in Brette & Gerstner (2005)
	const Potential BG_E_L   = -70.6;  //mV; leak potential

	//! Threshold potential in Brette & Gerstner (2005)
	const Potential BG_V_T   = -50.4;  //mV; threshold potential

	//! Reset potential in Brette & Gerstner (2005)
	const Potential BG_V_R   =  BG_E_L; //equal to leak potential

	//! Slope parameter in Brette & Gerstner (2005)
	const Potential BG_D_T   = 2;       //mV; slope parameter

	//! Adaptation time constant in Brette & Gerstner (2005)
	const Time  BG_T_W      = 144;     //ms; adaptation constant

	//! Subthreshold adaptation in Brette & Gerstner (2005)
	const Conductance BG_A  = 4;	   //nS; subthreshold adaptation

	//! Spike-triggered adaptation in Brette & Gerstner (2005)
	const Current     BG_B  = 0.0805;  //nA; spike-triggered adaptation

	//! default maximum term size for AbstractNonCirculantSolver
	const double EPS_J_CIRC_MAX = 1e-10;

	//! Even if refraction is not considered, for some algorithms it is convenient to set it artificially to a non zero value
	const Time TIME_REFRACT_MIN = 0.1e-3;

	//! Standard start time of simulation
	const Time T_START = 0.0;

	//! Default time step
	const Time T_STEP = 1e-6;

	//! Default precision for running numerical integrations
	const Precision PRECISION(1e-7,0.0);

	//! Maximum number of iterations before DynamicNetwork reports a failure
	const Number MAXITERATION = 10000000;

	//! Maximum number of terms that the DiffusionMasterEquation should take into account
	const Number N_DIFF_MAX  = 20;

	//! Batch times for probability queues
	const Time TIME_QUEUE_BATCH = 1e-6;
	*/
} // end of GeomLib

#endif // include guard
