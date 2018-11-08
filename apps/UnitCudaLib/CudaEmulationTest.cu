#include <boost/timer/timer.hpp>
#include "TwoDLib.hpp"
#include "CudaTest.cuh"
#include "MPILib/include/DelayedConnection.hpp"


#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
//
class CudaAlgorithm {
public:

   CudaAlgorithm
   (
    const TwoDLib::Ode2DSystem&, 
    const TwoDLib::TransitionMatrix& mat,
    const std::vector<TwoDLib::Redistribution>& vec_reversal,
    const std::vector<TwoDLib::Redistribution>& vec_reset
);
   void Configure(fptype t_end, unsigned int n_div);
   ~CudaAlgorithm();

private:

   const TwoDLib::Ode2DSystem& _sys;
   unsigned int _step;
   unsigned int _nr_steps;
   fptype  _t_step;
   TwoDLib:: CSRMatrix _mat;

public:
  // the arrays in the section need to be put on the device

   // these hold the data and derivatives
   fptype* _mass;
   fptype* _derivative;

   // this is an unsigned integer to label the first strip number of strip 1
   unsigned int* _i_1;

   // these are required to represent the CSR matrix
   unsigned int _nnz;
   unsigned int _nia;
   fptype* _val;
   unsigned int* _ia;
   unsigned int* _ja;

  // these are required to represent the mapping
   unsigned int* _length;
   unsigned int* _first;
   unsigned int* _map;

  // reversal mapping
  unsigned int  _n_reversal;
  unsigned int* _rev_from;
  unsigned int* _rev_to;
  fptype*        _rev_alpha;

  // reversal mapping
  unsigned int  _n_reset;
  unsigned int* _res_from;
  unsigned int* _res_to;
  fptype*       _res_alpha;

  // rate variable
  fptype*      _rate; 
};

CudaAlgorithm::CudaAlgorithm
(
 const TwoDLib::Ode2DSystem& sys, 
 const TwoDLib::TransitionMatrix& mat,
 const std::vector<TwoDLib::Redistribution>& map_reversal,
 const std::vector<TwoDLib::Redistribution>& map_reset
)
:
_sys(sys),
_step(0),
_nr_steps(0),
_t_step(sys.MeshObject().TimeStep()),
_mat(mat,sys),
_mass(0),
_derivative(0),
_nnz(0)
{
    unsigned int N = sys.Mass().size(); 
   _nnz = _mat.Val().size(); // number of non-zeros
   _nia = _mat.Ia().size();  // number of rows + 1 

   cudaMallocManaged(&_mass,N*sizeof(fptype));
   cudaMallocManaged(&_derivative,N*sizeof(fptype));
   cudaMallocManaged(&_i_1,sizeof(unsigned int));
   *_i_1 = sys.MeshObject().NrCellsInStrip(0) + 1; // record the first non-stationary bin, remap must start from here
 
    // absorb arrays from the CSRMatrix object, with the intent to pass them on to the device
    cudaMallocManaged(&_val,_nnz*sizeof(fptype));
    for (unsigned int i = 0; i < _nnz; i++)
       _val[i] = _mat.Val()[i];

    cudaMallocManaged(&_ia,_nia*sizeof(unsigned int));
    for (unsigned int i = 0; i < _nia; i++)
       _ia[i] = _mat.Ia()[i];

    cudaMallocManaged(&_ja,_nnz*sizeof(unsigned int));
    for (unsigned i = 0; i < _nnz; i++)
       _ja[i] = _mat.Ja()[i];


   // initialize the mapping arrays

    cudaMallocManaged(&_first, N*sizeof(unsigned int));
    cudaMallocManaged(&_length,N*sizeof(unsigned int));
    cudaMallocManaged(&_map,   N*sizeof(unsigned int));

    std::cout << "Mapping arrays created" << std::endl;
    unsigned int counter = 0;
    const TwoDLib::Mesh& mesh = sys.MeshObject();
    for (unsigned int i = 0; i < mesh.NrQuadrilateralStrips(); i++){
      unsigned int first = counter;
      for (unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++){
        _first[counter]  = first;
        _length[counter] = mesh.NrCellsInStrip(i);
        _map[counter]    = counter;
        counter++;
     }
   }
   // reversal mapping
   _n_reversal = map_reversal.size();
   cudaMallocManaged(&_rev_from, _n_reversal*sizeof(unsigned int));
   cudaMallocManaged(&_rev_to,   _n_reversal*sizeof(unsigned int));
   cudaMallocManaged(&_rev_alpha,_n_reversal*sizeof(fptype));

   counter = 0;
   for( auto r: map_reversal)
   {
     _rev_from[counter]  =  sys.Map(r._from[0],r._from[1]);
     _rev_to[counter]    =  sys.Map(r._to[0],r._to[1]);
     _rev_alpha[counter] =  r._alpha;

     counter++;
   }

   // reversal mapping                                                                                                                                                               
   _n_reset = map_reset.size();
   cudaMallocManaged(&_res_from,  _n_reset*sizeof(unsigned int));
   cudaMallocManaged(&_res_to,    _n_reset*sizeof(unsigned int));
   cudaMallocManaged(&_res_alpha, _n_reset*sizeof(fptype));

   counter = 0;
   for( auto r: map_reset)
     {
       _res_from[counter]  =  sys.Map(r._from[0],r._from[1]);
       _res_to[counter]    =  sys.Map(r._to[0],r._to[1]);
       _res_alpha[counter] =  r._alpha;

       counter++;
     }
   cudaMallocManaged(&_rate, sizeof(float));
   *_rate = 0.;
}


void CudaAlgorithm::Configure(fptype t_end, unsigned int n_div)
{
   _nr_steps = static_cast<unsigned int>(ceil(t_end/_t_step));
   _step = 0;

   int N = _sys.Mass().size();
   for(int i = 0; i < N; i++)
     _mass[i] = 0.;
   
   _mass[0] = 1.;
   
   for(int i = 0; i < N; i++)
     _derivative[i] = 0.; 
}


CudaAlgorithm::~CudaAlgorithm()
{
   cudaFree(_mass);
   cudaFree(_derivative);
   cudaFree(_i_1);
   cudaFree(_val);
   cudaFree(_ia);
   cudaFree(_ja);
   cudaFree(_length);
   cudaFree(_first);
   cudaFree(_map);
   cudaFree(_rev_from);
   cudaFree(_rev_to);
   cudaFree(_rev_alpha);
   cudaFree(_res_from);
   cudaFree(_res_to);
   cudaFree(_res_alpha);
}


fptype sum(fptype* f, unsigned int N)
{
   fptype sum = 0.;
   for (unsigned int i = 0; i < N; i++)
       sum += f[i];
   return sum;
}

void Dump(const std::string& fn, const TwoDLib::Mesh& mesh, fptype* mass, unsigned int* map){
    std::ofstream ofst(fn);
    unsigned int count = 0;
    for (unsigned int i = 0; i < mesh.NrQuadrilateralStrips(); i++)
        for(unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++){
            ofst << i << '\t' << j << '\t' << mass[map[count++]]/fabs(mesh.Quad(i,j).SignedArea()) << '\t';
        }
}

int main()
{
  
  std::string strmat("fn551e96aa-7996-43c7-a77e-c92f7b90efa8_0.02_0_0_0_.mat"); 
  TwoDLib::TransitionMatrix mat(strmat);
  std::vector<std::string> vecmat;
  vecmat.push_back(strmat);
  std::string strmod("fn551e96aa-7996-43c7-a77e-c92f7b90efa8.model");
  // not used, because we set the integration time step by setting n_div below,
  // but an argument must be provided
  double t_int_step = 1e-5;
  TwoDLib::MeshAlgorithm<MPILib::DelayedConnection> alg(strmod,vecmat,t_int_step);
  const TwoDLib::Ode2DSystem& sys = alg.Sys();

  std::cout <<  "Creating Algorithm" << std::endl;
  CudaAlgorithm cualg(sys,mat,alg.ReversalMap(),alg.ResetMap());
  std::cout <<  "Algorithm created"  << std::endl;
  unsigned int N = alg.Sys().Mass().size();
  
  fptype t_step = alg.Sys().MeshObject().TimeStep();  
  // simulation end time 
  fptype t_end = 100.0;
  // calculate the total number of steps
  unsigned int total_step = static_cast<unsigned int>(ceil(t_end/t_step)); 
  // divide the mesh steps to arrive at integration
  unsigned int n_div = 10;
  // input firing rate
  fptype rate = 2.;
  cualg.Configure(t_end,n_div);
  std::cout << "Starting" << std::endl;
  boost::timer::auto_cpu_timer t; 
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  for (unsigned int step = 0; step < total_step; step++){  
  //blockSize = 1;
  //numBlocks = 1;
     Remap<<<numBlocks,blockSize>>>(N,cualg._i_1,step,cualg._map,cualg._first,cualg._length);
     for (unsigned int i = 0; i < n_div; i++){
         CalculateDerivative<<<numBlocks,blockSize>>>(N,cualg._derivative,cualg._mass,cualg._val,cualg._ia,cualg._ja,cualg._map);
         EulerStep<<<numBlocks,blockSize>>>(N,n_div,cualg._derivative,cualg._mass,t_step,rate);
     } 
     MapReversal<<<1,1>>>(cualg._n_reversal,cualg._rev_from,cualg._rev_to,cualg._rev_alpha,cualg._mass,cualg._map);
     MapReset<<<1,1>>>   (cualg._n_reset   ,cualg._res_from,cualg._res_to,cualg._res_alpha,cualg._mass,cualg._map,cualg._rate);
     MapFinish<<<1,1>>>(cualg._n_reset,cualg._res_from,cualg._mass,cualg._map);
     cudaDeviceSynchronize();
     //std::cout << (*cualg._rate)/t_step << std::endl;
  }


  std::cout << "Overall time spend\n";
  t.report();
  Dump(std::string("mesh"),alg.Sys().MeshObject(),cualg._mass,cualg._map);

  std::cout << cualg._mass[0] << " " << sum(cualg._mass,N) << std::endl;
 
  return 0;

}
