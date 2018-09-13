#include "TwoDLib.hpp"

#include "MPILib/include/DelayedConnection.hpp"

class CudaAlgorithm {
public:

   CudaAlgorithm
   (
    const TwoDLib::Ode2DSystem&, 
    const TwoDLib::TransitionMatrix& mat,
    const std::vector<TwoDLib::Redistribution>& vec_reversal,
    const std::vector<TwoDLib::Redistribution>& vec_reset
);
   void Configure(float t_end, unsigned int n_div);
   ~CudaAlgorithm();

private:

   const TwoDLib::Ode2DSystem& _sys;
   unsigned int _step;
   unsigned int _nr_steps;
   float  _t_step;
   TwoDLib:: CSRMatrix _mat;

public:
  // the arrays in the section need to be put on the device

   // these hold the data and derivatives
   float* _mass;
   float* _derivative;

   // these are required to represent the CSR matrix
   unsigned int _nnz;
   unsigned int _nia;
   float* _val;
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
  float*        _rev_alpha;

  // reversal mapping
  unsigned int  _n_reset;
  unsigned int* _res_from;
  unsigned int* _res_to;
  float*        _res_alpha;
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

   _mass       = (float*)malloc(N*sizeof(float)); // mass array
   _derivative = (float*)malloc(N*sizeof(float)); // derivative in Euler step
 
  // absorb arrays from the CSRMatrix object, with the intent to pass them on to the device
  _val =        (float*)malloc(_nnz*sizeof(float));
    for (unsigned int i = 0; i < _nnz; i++)
       _val[i] = _mat.Val()[i];
   _ia = (unsigned int*)malloc(_nia*sizeof(unsigned int));
   for (unsigned int i = 0; i < _nia; i++)
       _ia[i] = _mat.Ia()[i];
   _ja = (unsigned int*)malloc(_nnz*sizeof(unsigned int));
   for (unsigned i = 0; i < _nnz; i++)
       _ja[i] = _mat.Ja()[i];

   // initialize the mapping arrays
   _first  = (unsigned int*)malloc(N*sizeof(unsigned int));
   _length = (unsigned int*)malloc(N*sizeof(unsigned int));
   _map    = (unsigned int*)malloc(N*sizeof(unsigned int));

   unsigned int counter = 0;
   const TwoDLib::Mesh& mesh = sys.MeshObject();
   for (unsigned int i = 0; i < mesh.NrStrips(); i++){
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
   _rev_from  = (unsigned int*)malloc(_n_reversal*sizeof(unsigned int));
   _rev_to    = (unsigned int*)malloc(_n_reversal*sizeof(unsigned int));
   _rev_alpha = (float*)malloc(_n_reversal*sizeof(float));

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
   _res_from  = (unsigned int*)malloc(_n_reset*sizeof(unsigned int));
   _res_to    = (unsigned int*)malloc(_n_reset*sizeof(unsigned int));
   _res_alpha = (float*)malloc(_n_reset*sizeof(float));

   counter = 0;
   for( auto r: map_reset)
     {
       _res_from[counter]  =  sys.Map(r._from[0],r._from[1]);
       _res_to[counter]    =  sys.Map(r._to[0],r._to[1]);
       _res_alpha[counter] =  r._alpha;

       counter++;
     }
}


void CudaAlgorithm::Configure(float t_end, unsigned int n_div)
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
   free(_mass);
   free(_derivative);
   free(_val);
   free(_ia);
   free(_ja);
   free(_length);
   free(_first);
   free(_map);
   free(_rev_from);
   free(_rev_to);
   free(_rev_alpha);
   free(_res_from);
   free(_res_to);
   free(_res_alpha);
}

void CalculateDerivative(unsigned int N, float* derivative, float* mass, float* val, unsigned int* ia, unsigned int* ja, unsigned int* map){
    for( unsigned int i = 0; i < N; i++){
      derivative[map[i]]  = -mass[map[i]];
      for(unsigned int j = ia[i]; j < ia[i+1]; j++)
	  derivative[map[i]] += val[j]*mass[map[ja[j]]];      
    }
}

void EulerStep(unsigned int N, unsigned int n_div, float* derivative, float* mass, float timestep, float rate)
{
    for (unsigned int i = 0; i < N; i++)
       mass[i] += derivative[i]*rate*timestep/static_cast<float>(n_div);
}

void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, float* rev_alpha, float* mass, unsigned int* map)
{
  for (int i = 0; i < n_reversal; i++){
    float m = mass[map[rev_from[i]]];
    mass[map[rev_from[i]]] = 0.;
    mass[map[rev_to[i]]] += m;
  }
}

void MapReset(unsigned int n_reset, unsigned int* res_from, unsigned int* res_to, float* res_alpha, float* mass, unsigned int* map, float* response)
{
  *response = 0.;
  for (int i = 0; i < n_reset; i++){
    float m = res_alpha[i]*mass[map[res_from[i]]];
    mass[map[res_to[i]]] += m;
    *response += m;
  }
  for (int i = 0; i < n_reset; i++)
    mass[map[res_from[i]]] = 0.;
 
}


void Remap(int N, unsigned int t, unsigned int *map, unsigned int* first, unsigned int* length)
{
    for (int i = 0; i < N; i++ )
      map[i] =  TwoDLib::modulo((i - t - first[i]),length[i]) + first[i];
}

float sum(float* f, unsigned int N)
{
   float sum = 0.;
   for (unsigned int i = 0; i < N; i++)
       sum += f[i];
   return sum;
}

void Dump(const std::string& fn, const TwoDLib::Mesh& mesh, float* mass, unsigned int* map){
    std::ofstream ofst(fn);
    unsigned int count = 0;
    for (unsigned int i = 0; i < mesh.NrStrips(); i++)
        for(unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++){
            ofst << i << '\t' << j << '\t' << mass[map[count++]]/fabs(mesh.Quad(i,j).SignedArea()) << '\t';
        }
}

int main()
{
  std::string strmat("condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat"); 
  TwoDLib::TransitionMatrix mat(strmat);
  std::vector<std::string> vecmat;
  vecmat.push_back(strmat);
  std::string strmod("condee2a5ff4-0087-4d69-bae3-c0a223d03693.model");
  // not used, because we set the integration time step by setting n_div below,
  // but an argument must be provided
  double t_int_step = 1e-5;
  TwoDLib::MeshAlgorithm<MPILib::DelayedConnection> alg(strmod,vecmat,t_int_step);
  const TwoDLib::Ode2DSystem& sys = alg.Sys();

  CudaAlgorithm cualg(sys,mat,alg.ReversalMap(),alg.ResetMap());
  unsigned int N = alg.Sys().Mass().size();
  
  float t_step = alg.Sys().MeshObject()s.TimeStep();  
  // simulation end time 
  float t_end = 1.0;
  // divide the mesh steps to arrive at integration
  unsigned int n_div = 10;
  // input firing rate
  float rate = 1000.;
  float response = 0.;
  cualg.Configure(t_end,n_div);
  std::cout << "Starting" << std::endl;
   for (unsigned int step = 0; step < 1000; step++){  
     Remap(N,step,cualg._map,cualg._first,cualg._length);
     for (unsigned int i = 0; i < n_div; i++){
         CalculateDerivative(N,cualg._derivative,cualg._mass,cualg._val,cualg._ia,cualg._ja,cualg._map);
         EulerStep(N,n_div,cualg._derivative,cualg._mass,t_step,rate);
     }
     MapReversal(cualg._n_reversal,cualg._rev_from,cualg._rev_to,cualg._rev_alpha,cualg._mass,cualg._map);
     MapReset   (cualg._n_reset   ,cualg._res_from,cualg._res_to,cualg._res_alpha,cualg._mass,cualg._map, &response);
     std::cout << response/t_step << std::endl;
  }
  Dump(std::string("mesh"),alg.Sys().MeshObject(),cualg._mass,cualg._map);

  std::cout << cualg._mass[0] << " " << sum(cualg._mass,N) << std::endl;
 
  return 0;

}
