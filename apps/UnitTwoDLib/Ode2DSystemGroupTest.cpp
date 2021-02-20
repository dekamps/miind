

// Copyright (c) 2005 - 2015 Marc de Kamps
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

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <TwoDLib.hpp>
#include "FixtureOde2DSystemGroup.hpp"

using namespace std;
using namespace TwoDLib;

BOOST_FIXTURE_TEST_CASE(SystemGroupConstructionTest,FixtureOde2DSystemGroup){

	std::vector<Redistribution> vec_dummy;
	std::vector<std::vector<Redistribution> > vec_vec_dummy;
	std::vector<Mesh> vec_mesh  {_mesh1,_mesh2};
	Ode2DSystemGroup sys(vec_mesh,vec_vec_dummy,vec_vec_dummy);
	BOOST_CHECK( sys.Map(0)  == 0 );
	BOOST_CHECK( sys.Map(1)  == 1 );
	BOOST_CHECK( sys.Map(2)  == 2 );
	BOOST_CHECK( sys.Map(3)  == 3 );
	BOOST_CHECK( sys.Map(4)  == 4 );
	BOOST_CHECK( sys.Map(5)  == 5 );
	BOOST_CHECK( sys.Map(6)  == 6 );
	BOOST_CHECK( sys.Map(7)  == 7 );
	BOOST_CHECK( sys.Map(8)  == 8 );
	BOOST_CHECK( sys.Map(9)  == 9 );
	BOOST_CHECK( sys.Map(10)  == 10 );
	BOOST_CHECK( sys.Map(11)  == 11 );
	BOOST_CHECK( sys.Map(12)  == 12);

	sys.Evolve();

	BOOST_CHECK( sys.Map(0)  == 0 );
	BOOST_CHECK( sys.Map(1)  == 1 );
	BOOST_CHECK( sys.Map(2)  == 3 );
	BOOST_CHECK( sys.Map(3)  == 2 );
	BOOST_CHECK( sys.Map(4)  == 6 );
	BOOST_CHECK( sys.Map(5)  == 4 );
	BOOST_CHECK( sys.Map(6)  == 5 );
	BOOST_CHECK( sys.Map(7)  == 7 );
	BOOST_CHECK( sys.Map(8)  == 10 );
	BOOST_CHECK( sys.Map(9)  == 8 );
	BOOST_CHECK( sys.Map(10)  == 9 );
	BOOST_CHECK( sys.Map(11)  == 12 );
	BOOST_CHECK( sys.Map(12)  == 11);

}

BOOST_FIXTURE_TEST_CASE(GroupMapTest, FixtureOde2DSystemGroup){


	std::vector<Redistribution> vec_dummy;
	std::vector<std::vector<Redistribution> > vec_vec_dummy;
	std::vector<Mesh> vec_mesh  {_mesh1, _mesh2, _mesh3};
	Ode2DSystemGroup sys(vec_mesh,vec_vec_dummy,vec_vec_dummy);

	BOOST_CHECK( sys.Map(0,0,0) == 0);
	BOOST_CHECK( sys.Map(0,0,1) == 1);
	BOOST_CHECK( sys.Map(0,1,0) == 2);
	BOOST_CHECK( sys.Map(0,1,1) == 3);
	BOOST_CHECK( sys.Map(0,2,0) == 4);
	BOOST_CHECK( sys.Map(0,2,1) == 5);
	BOOST_CHECK( sys.Map(0,2,2) == 6);
	BOOST_CHECK( sys.Map(1,0,0) == 7);
	BOOST_CHECK( sys.Map(1,1,0) == 8);
	BOOST_CHECK( sys.Map(1,1,1) == 9);
	BOOST_CHECK( sys.Map(1,1,2) == 10);
	BOOST_CHECK( sys.Map(1,2,0) == 11);
	BOOST_CHECK( sys.Map(1,2,1) == 12);
    BOOST_CHECK( sys.Map(2,0,0) == 13);
    BOOST_CHECK( sys.Map(2,0,1) == 14);
    BOOST_CHECK( sys.Map(2,1,0) == 15);
    BOOST_CHECK( sys.Map(2,1,1) == 16);
    BOOST_CHECK( sys.Map(2,2,0) == 17);
    BOOST_CHECK( sys.Map(2,2,1) == 18);
    BOOST_CHECK( sys.Map(2,2,2) == 19);


	sys.Evolve();
	BOOST_CHECK( sys.Map(0,0,0) == 0);
	BOOST_CHECK( sys.Map(0,0,1) == 1);
	BOOST_CHECK( sys.Map(0,1,0) == 3);
	BOOST_CHECK( sys.Map(0,1,1) == 2);
	BOOST_CHECK( sys.Map(0,2,0) == 6);
	BOOST_CHECK( sys.Map(0,2,1) == 4);
	BOOST_CHECK( sys.Map(0,2,2) == 5);
	BOOST_CHECK( sys.Map(1,0,0) == 7);
	BOOST_CHECK( sys.Map(1,1,0) == 10);
	BOOST_CHECK( sys.Map(1,1,1) == 8);
	BOOST_CHECK( sys.Map(1,1,2) == 9);
	BOOST_CHECK( sys.Map(1,2,0) == 12);
	BOOST_CHECK( sys.Map(1,2,1) == 11);
	BOOST_CHECK( sys.Map(2,0,0) == 13);
	BOOST_CHECK( sys.Map(2,0,1) == 14);
	BOOST_CHECK( sys.Map(2,1,0) == 16);
	BOOST_CHECK( sys.Map(2,1,1) == 15);
	BOOST_CHECK( sys.Map(2,2,0) == 19);
	BOOST_CHECK( sys.Map(2,2,1) == 17);
	BOOST_CHECK( sys.Map(2,2,2) == 18);

}

BOOST_AUTO_TEST_CASE(ConductanceMapTest){

	Mesh mesh("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh");
	Stat stat("condee2a5ff4-0087-4d69-bae3-c0a223d03693.stat");

	std::vector<Quadrilateral> vecq = stat.Extract();
	mesh.InsertStationary(vecq[0]);

	std::vector<Redistribution> vec_dummy;
	std::vector<std::vector<Redistribution> > vec_vec_dummy;
	std::vector<Mesh> vec_mesh{mesh};
	Ode2DSystemGroup sys(vec_mesh,vec_vec_dummy,vec_vec_dummy);

	BOOST_CHECK( sys.Map(0) == 0);
	BOOST_CHECK( sys.Map(1) == 1);
	BOOST_CHECK( sys.Map(2) == 2);
	BOOST_CHECK( sys.Map(3) == 3);

	sys.Evolve();


	BOOST_CHECK( sys.Map(0) == 0);
	BOOST_CHECK( sys.Map(1) == 700);
	BOOST_CHECK( sys.Map(2) == 1);
	BOOST_CHECK( sys.Map(3) == 2);

	sys.Evolve();

	BOOST_CHECK( sys.Map(0) == 0);
	BOOST_CHECK( sys.Map(1) == 699);
	BOOST_CHECK( sys.Map(2) == 700);
	BOOST_CHECK( sys.Map(3) == 1);
}

