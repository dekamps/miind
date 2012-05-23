#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <PopulistLib.h>

using namespace PopulistLib;

using namespace std;

BOOST_AUTO_TEST_SUITE( CharacteristicAlgorithmTest )

BOOST_AUTO_TEST_CASE( Creation )
{
	CharacteristicParameter par;
	CharacteristicAlgorithm<double> alg(par);

	BOOST_REQUIRE( true );
}

BOOST_AUTO_TEST_CASE( Copy )
{
	CharacteristicParameter par;
	CharacteristicAlgorithm<double> alg(par);	

	CharacteristicParameter pardif;
	CharacteristicAlgorithm<double> algdif(pardif);

	algdif = alg;

	BOOST_REQUIRE( true );
}



BOOST_AUTO_TEST_SUITE_END()
