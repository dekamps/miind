// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#include <vector>
#include <MPILib/include/TypeDefinitions.hpp>
#define private public
#define protected public
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>
#undef protected
#undef private
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::populist::parameters;
using namespace MPILib;


void test_Constructor() {

	PopulistSpecificParameter test;

	BOOST_CHECK(test._v_min == 0.0);
	BOOST_CHECK(test._n_grid_initial == 0);
	BOOST_CHECK(test._n_add == 0);
	BOOST_CHECK(test._par_dens._mu == 0.0);
	BOOST_CHECK(test._par_dens._sigma == 0.0);
	BOOST_CHECK(test._fact_expansion == 0.0);
	BOOST_CHECK(test._name_zeroleak == "");
	BOOST_CHECK(test._name_circulant == "");
	BOOST_CHECK(test._name_noncirculant == "");
	BOOST_CHECK(test._p_rebinner.get() != nullptr);
	BOOST_CHECK(test._p_rate.get() != nullptr);

	PopulistSpecificParameter test1(1.0, 2, 3,
			InitialDensityParameter(4.0, 5.0), 6.0);

	BOOST_CHECK(test1._v_min == 1.0);
	BOOST_CHECK(test1._n_grid_initial == 2);
	BOOST_CHECK(test1._n_add == 3);
	BOOST_CHECK(test1._par_dens._mu == 4.0);
	BOOST_CHECK(test1._par_dens._sigma == 5.0);
	BOOST_CHECK(test1._fact_expansion == 6.0);
	BOOST_CHECK(
			test1._name_zeroleak == std::string("NumericalZeroLeakEquations"));
	BOOST_CHECK(test1._name_circulant == std::string("CirculantSolver"));
	BOOST_CHECK(test1._name_noncirculant == std::string("NonCirculantSolver"));
	BOOST_CHECK(test1._p_rebinner.get() != nullptr);
	BOOST_CHECK(test1._p_rate.get() != nullptr);

	PopulistSpecificParameter test2(1.0, 2, 3,
			InitialDensityParameter(4.0, 5.0), 6.0, "blub1", "blub2", "blub3");
	BOOST_CHECK(test2._v_min == 1.0);
	BOOST_CHECK(test2._n_grid_initial == 2);
	BOOST_CHECK(test2._n_add == 3);
	BOOST_CHECK(test2._par_dens._mu == 4.0);
	BOOST_CHECK(test2._par_dens._sigma == 5.0);
	BOOST_CHECK(test2._fact_expansion == 6.0);
	BOOST_CHECK( test2._name_zeroleak == std::string("blub1"));
	BOOST_CHECK(test2._name_circulant == std::string("blub2"));
	BOOST_CHECK(test2._name_noncirculant == std::string("blub3"));
	BOOST_CHECK(test2._p_rebinner.get() != nullptr);
	BOOST_CHECK(test2._p_rate.get() != nullptr);
}

void test_CopyAndClone() {

	PopulistSpecificParameter test(1.0, 2, 3, InitialDensityParameter(4.0, 5.0),
			6.0, "blub1", "blub2", "blub3");

	PopulistSpecificParameter test1(test);
	BOOST_CHECK(test1._v_min == 1.0);
	BOOST_CHECK(test1._n_grid_initial == 2);
	BOOST_CHECK(test1._n_add == 3);
	BOOST_CHECK(test1._par_dens._mu == 4.0);
	BOOST_CHECK(test1._par_dens._sigma == 5.0);
	BOOST_CHECK(test1._fact_expansion == 6.0);
	BOOST_CHECK( test1._name_zeroleak == std::string("blub1"));
	BOOST_CHECK(test1._name_circulant == std::string("blub2"));
	BOOST_CHECK(test1._name_noncirculant == std::string("blub3"));
	BOOST_CHECK(test1._p_rebinner.get() != test._p_rebinner.get());
	BOOST_CHECK(test1._p_rate.get() != test._p_rate.get());

	PopulistSpecificParameter test2 = test;
	BOOST_CHECK(test2._v_min == 1.0);
	BOOST_CHECK(test2._n_grid_initial == 2);
	BOOST_CHECK(test2._n_add == 3);
	BOOST_CHECK(test2._par_dens._mu == 4.0);
	BOOST_CHECK(test2._par_dens._sigma == 5.0);
	BOOST_CHECK(test2._fact_expansion == 6.0);
	BOOST_CHECK( test2._name_zeroleak == std::string("blub1"));
	BOOST_CHECK(test2._name_circulant == std::string("blub2"));
	BOOST_CHECK(test2._name_noncirculant == std::string("blub3"));
	BOOST_CHECK(test2._p_rebinner.get() != test._p_rebinner.get());
	BOOST_CHECK(test2._p_rate.get() != test._p_rate.get());
	PopulistSpecificParameter* test3 = test.Clone();
	BOOST_CHECK(test3->_v_min == 1.0);
	BOOST_CHECK(test3->_n_grid_initial == 2);
	BOOST_CHECK(test3->_n_add == 3);
	BOOST_CHECK(test3->_par_dens._mu == 4.0);
	BOOST_CHECK(test3->_par_dens._sigma == 5.0);
	BOOST_CHECK(test3->_fact_expansion == 6.0);
	BOOST_CHECK( test3->_name_zeroleak == std::string("blub1"));
	BOOST_CHECK(test3->_name_circulant == std::string("blub2"));
	BOOST_CHECK(test3->_name_noncirculant == std::string("blub3"));
	BOOST_CHECK(test3->_p_rebinner.get() != test._p_rebinner.get());
	BOOST_CHECK(test3->_p_rate.get() != test._p_rate.get());

	delete test3;

}

void test_Getters(){

	PopulistSpecificParameter test1(1.0, 2, 3, InitialDensityParameter(4.0, 5.0),
			6.0, "blub1", "blub2", "blub3");
	BOOST_CHECK(test1._v_min == test1.getVMin());
	BOOST_CHECK(test1._n_grid_initial == test1.getNrGridInitial());
	BOOST_CHECK(test1._n_add == test1.getNrAdd());
	BOOST_CHECK(test1._par_dens._mu == test1.getInitialDensity()._mu);
	BOOST_CHECK(test1._par_dens._sigma == test1.getInitialDensity()._sigma);
	BOOST_CHECK(test1._fact_expansion == test1.getExpansionFactor());
	BOOST_CHECK( test1._name_zeroleak == test1.getZeroLeakName());
	BOOST_CHECK(test1._name_circulant == test1.getCirculantName());
	BOOST_CHECK(test1._name_noncirculant == test1.getNonCirculantName());
	BOOST_CHECK(test1._p_rebinner.get() == &test1.getRebin());
	BOOST_CHECK(test1._p_rate.get() == &test1.getRateComputation());
}

int test_main(int argc, char* argv[]) // note the name!
		{


	test_Constructor();
	test_CopyAndClone();
	test_Getters();
	return 0;
//    // six ways to detect and report the same error:
//    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error
//    BOOST_CHECK( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
