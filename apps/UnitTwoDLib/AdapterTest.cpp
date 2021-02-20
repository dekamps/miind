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
#include <TwoDLib.hpp>
#include <fstream>

using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(AdapterTest)
{

pugi::xml_document doc;
std::vector<TwoDLib::Mesh> vec_vec_mesh;
std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_rev;
std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_res;
pugi::xml_parse_result result0 = doc.load_file("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");
pugi::xml_node  root0 = doc.first_child();
TwoDLib::Mesh mesh0 = TwoDLib::RetrieveMeshFromXML(root0);
std::vector<TwoDLib::Mesh> vec_mesh{ mesh0, mesh0, mesh0 };


std::vector<TwoDLib::Redistribution> vec_reversal = TwoDLib::RetrieveMappingFromXML("Reversal",root0);
std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_reversal { vec_reversal, vec_reversal, vec_reversal};
std::vector<TwoDLib::Redistribution> vec_reset = TwoDLib::RetrieveMappingFromXML("Reset",root0);
std::vector< std::vector<TwoDLib::Redistribution> > vec_vec_reset { vec_reset, vec_reset, vec_reset};

TwoDLib::Ode2DSystemGroup group(vec_mesh, vec_vec_reversal, vec_vec_reset);
group.Initialize(0,0,0);
group.Initialize(1,0,0);
group.Initialize(2,0,0);

TwoDLib::TransitionMatrix mat("condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat");
TwoDLib::CSRMatrix csrmat1(mat,group,0);
TwoDLib::CSRMatrix csrmat2(mat,group,1);
TwoDLib::CSRMatrix csrmat3(mat,group,2);
std::vector<TwoDLib::CSRMatrix> vecmat{csrmat1, csrmat2, csrmat3};

TwoDLib::CSRAdapter csr_adapter(group,vecmat,1e-5);

std::vector<MPILib::Rate> vecrates { 800., 900., 1000. };
std::ofstream fs("fs.dat");

 for( int i = 0; i < 10; i++)
   {
     group.Evolve();
     group.RemapReversal();
     for (MPILib::Index i_part = 0; i_part < csr_adapter.NrIterations(); i_part++ ){
       csr_adapter.ClearDerivative();
       csr_adapter.CalculateDerivative(vecrates);
       csr_adapter.AddDerivative();
     }
     group.RedistributeProbability();
     // no MapFinish() in C++ version

     fs << group.F()[0] << " " <<  group.F()[1] <<  " " << group.F()[2] << '\n';
   }


 std::ofstream dumpdata1("group1simulation.data");
 std::ofstream dumpdata2("group2simulation.data");
 std::ofstream dumpdata3("group3simulation.data");
 std::vector<std::ostream*> vec_stream { &dumpdata1, &dumpdata2, &dumpdata3 };
 group.Dump(vec_stream);
}

