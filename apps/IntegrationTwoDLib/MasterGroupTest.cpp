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

#include <fstream>
#include <TwoDLib.hpp>

int main(int argc, char** argv)
{
  TwoDLib::Mesh mesh1("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");
  TwoDLib::Mesh mesh2("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");

  std::ifstream ifstrev("condee2a5ff4-0087-4d69-bae3-c0a223d03693.rev");
  std::vector<TwoDLib::Redistribution> vec_rev1 = TwoDLib::ReMapping(ifstrev);
  std::vector<TwoDLib::Redistribution> vec_rev2 = vec_rev1;
  
  std::ifstream ifstres("condee2a5ff4-0087-4d69-bae3-c0a223d03693.res");
  std::vector<TwoDLib::Redistribution> vec_res1 = TwoDLib::ReMapping(ifstres);
  std::vector<TwoDLib::Redistribution> vec_res2 = vec_res1;

  std::vector<TwoDLib::Mesh> mesh_vec{ mesh1, mesh2 };
  std::vector<std::vector<TwoDLib::Redistribution> > vec_rev{ vec_rev1, vec_rev2 };
  std::vector<std::vector<TwoDLib::Redistribution> > vec_res{ vec_res1, vec_res2 };
  
  TwoDLib::Ode2DSystemGroup group(mesh_vec,vec_rev,vec_res);
  group.Initialize(0,0,0);
  group.Initialize(1,0,0);
 
  TwoDLib::TransitionMatrix mat1("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mat");
  TwoDLib::TransitionMatrix mat2("condee2a5ff4-0087-4d69-bae3-c0a223d03693.mat");

  // this vector is required for the concatenation. It should not be used to create the Master equation
  // which is vector of different synaptic efficacies.
  std::vector<TwoDLib::TransitionMatrix> vec_mat {mat1, mat2};

  // therefore we create a new vector
  const std::vector<std::vector<TwoDLib::TransitionMatrix> > vectrans{vec_mat};
  TwoDLib::MasterParameter par(10);
  TwoDLib::MasterOdeint master(group, vectrans, par);

  MPILib::Number n_iter = 350; // this is sufficient for one threshold crossing in mesh 1
  for (int i = 0; i < n_iter; i++){
    group.Evolve();
    group.RemapReversal();
    group.RedistributeProbability();
  }

  std::ofstream ofst1("dens1.dat");
  std::ofstream ofst2("dens2.dat");

  std::vector<std::ostream*> vec_stream{ &ofst1, &ofst2 };
  group.Dump(vec_stream);
  return 0;
}
