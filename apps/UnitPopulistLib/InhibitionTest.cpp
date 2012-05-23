#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <PopulistLib.h>

using namespace PopulistLib;

void TestInhibition(CirculantMode mode){
	InputParameterSet input_set;

	for (int nr_bins = 2; nr_bins < 10; nr_bins++)
		for (int H = 1; H < nr_bins; H++)
			for (int i_init = 0; i_init < nr_bins; i_init++ )
				for ( Time tau = 0.0; tau < 5; tau += 0.05 )
				{
					int remainder = (nr_bins%H == 0) ? 0 : 1;
					int nr_n_c = nr_bins/H + remainder;

					valarray<double> test(0.0,nr_bins);
					test[i_init] = 1.0;

					NonCirculantSolver solver(mode);
					input_set._n_noncirc_exc = nr_n_c;
					input_set._H_exc = H;
					input_set._h_exc = H;
					input_set._alpha_exc = 0.0;

					solver.Configure(test, input_set);
					solver.ExecuteExcitatory
					(
						nr_bins,
						tau
					);


					valarray<double> test_reverse(0.0,nr_bins);
					test_reverse[nr_bins - 1 - i_init] = 1.0;

					input_set._H_inh = H;
					input_set._h_inh = H;
					input_set._alpha_inh = 0.0;
					input_set._n_noncirc_inh = nr_n_c;

					solver.Configure(test_reverse, input_set);
					solver.ExecuteInhibitory
					(
						nr_bins,
						tau
					);


					for (int i = 0; i < nr_bins; i++ )
						BOOST_CHECK( test[i] == test_reverse[nr_bins - 1 - i] );
				}
}

BOOST_AUTO_TEST_CASE(InhibitiontestIntegral){
	TestInhibition(INTEGER);
}

BOOST_AUTO_TEST_CASE(InihibitionTestFloatingPoint){
	TestInhibition(FLOATING_POINT);
}