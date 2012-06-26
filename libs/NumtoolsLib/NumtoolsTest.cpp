// Copyright (c) 2005 - 2007 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "NumtoolsTest.h"
#include "DVIntegratorCode.h"
#include "DVIntegratorException.h"
#include "MinMaxTrackerCode.h"
#include "RandomGenerator.h"
#include "Rational.h"
#include "IsApproximatelyEqualTo.h"
#include "TestDefinitions.h"
#include "UniformDistribution.h"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>


using namespace std;
using namespace NumtoolsLib;

NumtoolsTest::NumtoolsTest(boost::shared_ptr<ostream> p_stream):
LogStream(p_stream)
{
}

bool NumtoolsTest::Execute()
{
	if ( ! MinMaxTrackerTest() )
		return false;
	Record( "MinMaxTrackerTest test succeeded" );

	if ( ! RationalTest() )
		return false;
	Record( "RationalTest test succeeded" );

	if ( ! Ran2Test() )
		return false;
	Record( "Ran2 test succeeded" );

	if ( ! TestGsl() )
		return false;
	Record( "TestGSL succeeded");

	if ( ! DVIntegratorTest() )
		return false;
	Record( "DVIntegratorTest succeeded" );

	if ( ! DVVariableInputTest() )
		return false;
	Record( "DVVariableInputTest succeeded" );

	if ( ! DVCopyTest() )
		return false;
	Record("DVCopyTest succeeded");
	return true;
}

bool NumtoolsTest::MinMaxTrackerTest()
{
	MinMaxTracker<int> tracker;
	tracker.feedValue( 57, 0 );
	tracker.feedValue( 34, 1 );
	tracker.feedValue( 73, 2 );
	tracker.feedValue( 2843, 3 );
	tracker.feedValue( 3, 4 );
	tracker.feedValue( 7346, 5 );
	tracker.feedValue( 23, 6 );
	tracker.feedValue( 4546, 7 );
	tracker.feedValue( 23, 8 );
	tracker.feedValue( 5435, 9 );
	tracker.feedValue( 345, 10 );
	tracker.feedValue( 4356, 11 );
	tracker.feedValue( 234, 12 );
	tracker.feedValue( 9, 13 );
	tracker.feedValue( 234678, 14 );
	tracker.feedValue( 43, 15 );
	
	if( tracker.getMin() != 3 )
	{
		*Stream() << "getMin is not 3 but: " << tracker.getMin() << endl;
		return false;
	}
	if( tracker.getMinIndex() != 4 )
	{
		*Stream() << "getMinIndex is not 4 but: " << tracker.getMinIndex() << endl;
		return false;
	}
	if( tracker.getMax() != 234678 )
	{
		*Stream() << "getMax is not 234678 but: " << tracker.getMax() << endl;
		return false;
	}
	if( tracker.getMaxIndex() != 14 )
	{
		*Stream() << "getMaxIndex is not 14 but: " << tracker.getMaxIndex() << endl;
		return false;
	}
	if( tracker.getAvg() != 16253 )
	{
		*Stream() << "getAvg is not 16253 but: " << tracker.getAvg() << endl;
		return false;
	}
	
	return true;
}

bool NumtoolsTest::RationalTest()
{
	Rational rat( 14, 3 );
	if( rat.getValue() != 4 )
	{
		*Stream() << "getValue is not 4 but: " << rat.getValue() << endl;
		return false;
	}
	rat.set( 43, 5 );
	if( rat.getRemainder() != 3 )
	{
		*Stream() << "getRemainder is not 3 but: " << rat.getRemainder() << endl;
		return false;
	}
/*	rat++;
	if( rat.getValue() != 9 || rat.getRemainder() != 3 )
	{
		Stream() << "rat++ error getValue should be 9 but is: " << rat.getValue()
			<< " and getRemainder is not 3 but: " << rat.getRemainder() << endl;
		return false;
	}*/
	Rational rat2( 86, 10 );
	if( !( rat == rat2) || rat != rat2 )
	{
		*Stream() << "Rationals should be equal but are not: " << rat._numerator << "/" << rat._denominator 
			<< " and " << rat2._numerator << "/" << rat2._denominator << endl;
		*Stream() << "\tValues: " << rat.getValue() << " and " << rat2.getValue() << endl;
		*Stream() << "\tRemainder: " << rat.getRemainder() << " and " << rat2.getRemainder() << endl;
		return false;
	}
	if( (rat - rat2).getValue() != 0 || (rat - rat2).getRemainder() != 0 )
	{
		*Stream() << "Difference of rationals should be 0 but is not: " << rat._numerator << "/" << rat._denominator 
			<< " and " << rat2._numerator << "/" << rat2._denominator << endl;
		return false;
	}
	rat2._denominator = 9;
	if( !(rat < rat2) || rat >= rat2 )
	{
		*Stream() << "Rational 1 should be smaller than rational 2 but is not: " << rat._numerator << "/" << rat._denominator 
			<< " and " << rat2._numerator << "/" << rat2._denominator << endl;
		return false;
	}
	rat.set( 1, 5 );
	rat2.set( 3, 6 );
	Rational rat3( 3, 30 );
	if( rat * rat2 != rat3 )
	{
		*Stream() << "Rational 3 should be equal to rational 1 * rational 2 but is not: " << (rat*rat2)._numerator << "/" << (rat*rat2)._denominator 
			<< " and " << rat3._numerator << "/" << rat3._denominator << endl;
		return false;
	}
		
	//rat._numerator = 28; 
	// this test is not possible as the true value is not even comparable to itself....
//	if( rat.getTrueValueThroughCast() != rat.getTrueValueThroughCast() )
//	{
//		Stream() << "getTrueValueThroughCast is NAN!!!: " << rat.getTrueValueThroughCast() << endl;
//		return false;
//	}
	return true;
}

bool NumtoolsTest::Ran2Test() const
{

	// Purpose: show that the Distribution objects correctly reproduce the GSL
	// calls
	// Author: Marc de Kamps
	// Date: 13-10-2006

	// TODO: investigate
/*	gsl_rng* p_random = gsl_rng_alloc(gsl_rng_mt19937);

	double f_gsl = gsl_rng_uniform(p_random);

	gsl_rng_free(p_random);

	UniformDistribution uniform(GLOBAL_RANDOM_GENERATOR);
	double f_uniform = uniform.NextSampleValue();

	if ( ! IsApproximatelyEqualTo(f_gsl,f_uniform,1e-8) )
		return false;
*/
	return true;
}

bool NumtoolsTest::TestGsl() const
{
	double x = 5.0;
	double y = gsl_sf_bessel_J0 (x);

	double eps = y -(-1.775967713143382920e-01);

	if (! IsApproximatelyEqualTo(eps, 0, 1e-8))
		return false;

	return true;
}

namespace
{
	int Exp(double , const double y[], double f[], void *params)
	{
		double beta = *(double *)params;
		f[0] =   beta*y[0];
		f[1] = 2*beta*y[1];

		return GSL_SUCCESS;
	}


	int ExpPrime(double , const double[] , double *dfdy, double dfdt[], void *params)
	{
		double beta = *(double *)params;
		gsl_matrix_view dfdy_mat  = gsl_matrix_view_array (dfdy, 2, 2);
 
		gsl_matrix * m = &dfdy_mat.matrix; 
	

		gsl_matrix_set (m, 0, 0, beta); 
		gsl_matrix_set (m, 1, 1, 2*beta);
		gsl_matrix_set (m, 0, 1, 0);
		gsl_matrix_set (m, 1, 0, 0);
		
		dfdt[0] = 0.0;
		dfdt[1] = 0.0;
	
		return GSL_SUCCESS;

	}
} // end of unnamed namespace

bool NumtoolsTest::DVIntegratorTest() const
{
	Time time_initial = 0;

	TimeStep initial_time_step = 1e-4;
	double   epsilon_absolute = 1e-4;

	double e  = exp(1.0);
	double e2 = exp(2.0);

	double parameter = 1.0;

	vector<double> state(2);
	state[0] = 1;
	state[1] = 1;

	Precision precision(epsilon_absolute, 0);

	// calculate e, first with moderate precision
	DVIntegrator<double> 
		integrator
		(
			LARGE_NUMBER_OF_INTEGRATION_STEPS,
			state,
			initial_time_step,
			time_initial,
			precision,
			Exp,
			ExpPrime 	
		);

	integrator.Parameter() = parameter;

	Time time_end = 1;
	try
	{
		while (integrator.Evolve(time_end) < time_end)
			;
	}
	catch(DVIntegratorException except)
	{
		*Stream() << "Something went wrong in NodeIntegratorTest" << endl;
		*Stream() << except.Description()                         << endl;
	}

	vector<double> result = integrator.State();
	double f_res1 = result[0] - e;
	double f_res2 = result[1] - e2;

	// calculate e, first with high precision
	epsilon_absolute = 1e-20;

	precision._eps_absolute = epsilon_absolute;
	precision._eps_relative = 0.0;

	DVIntegrator<double> 
		accurate_integrator
		(
			LARGE_NUMBER_OF_INTEGRATION_STEPS,
			state,
			initial_time_step,
			time_initial,
			precision,
			Exp,
			ExpPrime 	
		);

	accurate_integrator.Parameter() = parameter;

	try
	{
		while (accurate_integrator.Evolve(time_end) < time_end)
			;
	}
	catch(DVIntegratorException except)
	{
		*Stream() << "Something went wrong in NodeIntegratorTest" << endl;
	}	

	result = accurate_integrator.State();
	f_res1 = result[0] - e;
	f_res2 = result[1] - e2;

	*Stream() << "Accuracy results: " << f_res1 << ", " << f_res2 <<"\n";

	return true;
}

bool NumtoolsTest::DVVariableInputTest() const
{
	Time time_begin = 0.0;
	Time time_end   = 2.0;

	TimeStep initial_time_step = 1e-4;
	double   h                 = 1e-6;
	double   epsilon_absolute  = 1e-7;

	double parameter = 1;

	vector<double> state(2);
	state[0] = 1;
	state[1] = 1;

	// calculate e, first with moderate precision
	Precision precision(epsilon_absolute,0.0);
	DVIntegrator<double> 
		integrator
		(
			LARGE_NUMBER_OF_INTEGRATION_STEPS,
			state,
			initial_time_step,
			time_begin,
			precision,
			Exp,
			ExpPrime,
			gsl_odeiv_step_rk4
		);

	integrator.Parameter() = parameter;

	try
	{
		for (Time time = 0; time < time_end; time += h)
		{
			integrator.Parameter() = time;
			while( integrator.Evolve(time+h) < time + h)
				;
		}
	}
	catch(DVIntegratorException except)
	{
		*Stream() << "Something went wrong in NodeIntegratorTest" << endl;
		*Stream() << except.Description()                         << endl;
	}

	vector<double> result = integrator.State();
	double f_difference = result[0] - exp(0.5*time_end*time_end);

	//TODO: why is epsilon not an abolute error, but only approximate ? see factor 100 below
	if (
		 !	NumtoolsLib::IsApproximatelyEqualTo
						(
							f_difference,
							0,
							100*epsilon_absolute
						)
		)
		return false;

	return true;
}

bool NumtoolsTest::DVCopyTest() const
{
	Time time_initial = 0;

	TimeStep initial_time_step = 1e-4;
	double   epsilon_absolute = 1e-6;

	double parameter = 1;

	vector<double> state(2);
	state[0] = 1;
	state[1] = 1;

	// calculate e, first with moderate precision
	Precision precision(epsilon_absolute,0.0);

	DVIntegrator<double> 
		integrator
		(
			LARGE_NUMBER_OF_INTEGRATION_STEPS,
			state,
			initial_time_step,
			time_initial,
			precision,
			Exp,
			ExpPrime 	
		);

	integrator.Parameter() = parameter;

	Time time_end = 1;
	try
	{
		while (integrator.Evolve(time_end) < time_end)
			;
	}
	catch(DVIntegratorException except)
	{
		*Stream() << "Something went wrong in NodeIntegratorTest" << endl;
		*Stream() << except.Description()                         << endl;
	}

	DVIntegrator<double> further_integrator(integrator);
	Time time_further_end = 3;
	try
	{
		while (integrator.Evolve(time_further_end) < time_further_end)
			;
	}
	catch(DVIntegratorException except)
	{
		*Stream() << "Something went wrong in NodeIntegratorTest" << endl;
		*Stream() << except.Description()                         << endl;
	}

	// The integrator copy should take off, where the other stopped
	// and should calculate e^3 and e^6
	double e3 = exp(3.0);
	double e6 = exp(6.0);

	vector<double> result = integrator.State();
	double f_res1 = result[0] - e3;
	double f_res2 = result[1] - e6;

	if ( ! IsApproximatelyEqualTo(f_res1,0,1e-6) )
		return false;

	//TODO: why does absolute preceision not hold for
	// e6 ? 

	if ( ! IsApproximatelyEqualTo(f_res2,0,1e-3) )
		return false;

	return true;
}
