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

#include <boost/test/unit_test.hpp>
#include <boost/circular_buffer.hpp>
#include <GeomLib.hpp>

using namespace GeomLib;

BOOST_AUTO_TEST_CASE(QIFOdeSystemTest)
{

    Number n_bins = 400;

    Time t_mem        =  10e-3;
    Time t_ref        =  0.0;
    Potential V_min   = -10.0;
    Potential V_peak  =  10.0;
    Potential V_reset = -10.0;
    Potential gamma   =  0.5;

    QifParameter
    par_qif
    (
        gamma
    );

	NeuronParameter par_neuron(V_peak, V_reset, V_reset, t_ref, t_mem);
    OdeParameter
    par_ode
    (
    	n_bins,
		V_min,
		par_neuron,
        InitialDensityParameter(V_reset, 0.0)
    );

    SpikingQifNeuralDynamics dyn(par_ode, par_qif);
    QifOdeSystem sys(dyn);

    vector<double> array_interpretation(n_bins, 0);
    vector<double> array_density(n_bins);

    double single_bin = 1.0;

    sys.PrepareReport(&array_interpretation[0], &array_density[0]);
    BOOST_CHECK_CLOSE(single_bin, array_density[0] * (array_interpretation[1] - array_interpretation[0]), 1e-3);
    BOOST_CHECK_CLOSE(par_ode._V_min, array_interpretation[0], 1e-3);
}

BOOST_AUTO_TEST_CASE(QIFEvolvePositive)
{

    Number n_bins = 10; // total number of bins
    Number n_shift = 5; // number of times times t_step, i.e. the number of bins that the density peak will be shifted.

    Time t_mem        =  10e-3;
    Time t_ref        =  0.0;
    Potential V_min   = -10.0;
    Potential V_peak  =  10.0;
    Potential V_reset = -10.0;
    Potential gamma   =  0.5;

    QifParameter
    par_qif
    (
        gamma
    );

	NeuronParameter par_neuron(V_peak, V_reset, V_reset, t_ref, t_mem);

    OdeParameter
    par_ode
    (
    	n_bins,
		V_min,
		par_neuron,
        InitialDensityParameter(V_reset, 0.0)
    );

    SpikingQifNeuralDynamics dyn(par_ode, par_qif);
    QifOdeSystem sys(dyn);

    vector<double> buffer_interpretation(n_bins, 0);
    vector<double> array_density(n_bins, 0);

    sys.PrepareReport(&buffer_interpretation[0], &array_density[0]);
    Time t = dyn.TimeToInf(par_ode._V_min) / n_bins;

    for (Index i = 0; i < n_shift; i++)
        sys.Evolve(t);

    sys.PrepareReport(&buffer_interpretation[0], &array_density[0]);

    // density was initialised at _V_reset; this has now shifted
    double sqr = sqrt(par_qif._gamma);
    double V_new = sqr * tan(sqr * (n_shift * t / par_ode._par_pop._tau) + atan(par_ode._V_min / sqr));

    
    BOOST_CHECK(fabs(V_new - buffer_interpretation[n_shift]) < 1e-10); // do not use BOOST_CHECK_CLOSE as it compares 1e-16 with 1e-17 and fails
    // to show that density is a density, convert back to probability
    BOOST_CHECK_CLOSE(1.0, array_density[n_shift] * (buffer_interpretation[n_shift + 1] - buffer_interpretation[n_shift]), 1e-3);
}

BOOST_AUTO_TEST_CASE(RateCalculationTest)
{
    Number n_bins = 10; // total number of bins

    Time t_mem   =  10e-3;
    Time t_ref   =  0.0;
    Potential V_min   = -10.0;
    Potential V_peak  =  10.0;
    Potential V_reset = -10.0;
    Potential gamma   =  0.5;

    QifParameter
    par_qif
    (
        gamma
    );

	NeuronParameter par_neuron(V_peak, V_reset, V_reset, t_ref, t_mem);
    // create a peak
    OdeParameter
    par_ode
    (
    	n_bins,
		V_min,
		par_neuron,
        InitialDensityParameter(V_reset, 0.0)
    );

    SpikingQifNeuralDynamics dyn(par_ode, par_qif);
    QifOdeSystem sys(dyn);

    vector<double> buffer_interpretation(n_bins, 0);
    vector<double> array_density(n_bins, 0);

    sys.PrepareReport(&buffer_interpretation[0], &array_density[0]);
    Time t = dyn.TimeToInf(par_ode._V_min) / n_bins;

    for (Index i = 0; i < n_bins - 1; i++)
        sys.Evolve(t);

    sys.PrepareReport(&buffer_interpretation[0], &array_density[0]);
    // all probability should now be shift in the highest bin and the firing rate should have been quiet throughout
    MPILib::Rate r_0 = sys.CurrentRate();

    BOOST_CHECK(r_0 == 0.0);

    // this should pust probability through threshold, causing a large peak in firing rate
    t =  dyn.TimeToInf(par_ode._V_min) / n_bins;
    sys.Evolve(t);
    sys.PrepareReport(&buffer_interpretation[0], &array_density[0]);
    MPILib::Rate r_1 = sys.CurrentRate();
    BOOST_CHECK_CLOSE(r_1 * dyn.TimeToInf(par_ode._V_min) / n_bins, 1.0, 1e-3);
}

