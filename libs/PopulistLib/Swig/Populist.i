%module Populist
%include "std_string.i"
%{
	#include "../UtilLib/UtilLib.h"
	#include "../NetLib/NetLib.h"
	#include "../DynamicLib/AbstractAlgorithm.h"
	#include "../DynamicLib/AbstractReportHandler.h"
	#include "../DynamicLib/AsciiReportHandler.h"
	#include "../DynamicLib/BasicDefinitions.h"
        #include "../DynamicLib/CanvasParameter.h"
	#include "../DynamicLib/DynamicNode.h"
	#include "../DynamicLib/DynamicNodeCode.h"
	#include "../DynamicLib/DynamicNetwork.h"
	#include "../DynamicLib/DynamicNetworkCode.h"
	#include "../DynamicLib/DynamicNetworkImplementation.h"
	#include "../DynamicLib/DynamicNetworkImplementationCode.h"
	#include "../DynamicLib/NodeInfo.h"
	#include "../DynamicLib/SimulationRunParameter.h"
	#include "../DynamicLib/RootReportHandler.h"
	#include "../DynamicLib/SpatialPosition.h"
	#include "../DynamicLib/WilsonCowanParameter.h"
	#include "../DynamicLib/WilsonCowanAlgorithm.h"
	#include "AbstractCirculantSolver.h"
	#include "AbstractNonCirculantSolver.h"
	#include "AbstractRebinner.h"
	#include "AbstractRateComputation.h"
	#include "BasicDefinitions.h"
   	#include "InitialDensityParameter.h"
	#include "IntegralRateComputation.h"
	#include "InterpolationRebinner.h"
	#include "OrnsteinUhlenbeckConnection.h"
	#include "OrnsteinUhlenbeckParameter.h"
	#include "PopulationAlgorithm.h"
	#include "PopulistParameter.h"
	#include "PopulistSpecificParameter.h"
	#include "ResponseParameterAmit.h"
	#include "ResponseParameterBrunel.h"
	#include "Response.h"
%}

namespace UtilLib {

	typedef unsigned int Number;
}

namespace NetLib {


	struct NodeId 
	{
		explicit NodeId():_id_value(0){}

		explicit NodeId(int id_value ):_id_value(id_value){}

		~NodeId(){}

		int _id_value;

	}; // end of NodeId

}


namespace DynamicLib { 

	typedef double Time;
	typedef double Rate;
	typedef double Density;
	typedef double Potential;

	struct CanvasParameter {

		double _t_min;
		double _t_max;
		double _state_min;
		double _state_max;
		double _f_max;
		double _dense_max;
	};

	class AbstractReportHandler
	{
	public:

		AbstractReportHandler
		(
			const std::string&
		);

		virtual ~AbstractReportHandler() = 0;



		std::string MediumName() const { return _stream_name; }

		virtual void  AddNodeToCanvas
		(
			NetLib::NodeId
		) const {}


	}; // end of AbstractReportmanager


	class RootReportHandler : public AbstractReportHandler
	{
	public:

		RootReportHandler
			(
				const  std::string&,
				bool   b_canvas,
				bool   b_force_state_write,
				const CanvasParameter&
			);


		virtual void AddNodeToCanvas(NetLib::NodeId) const;

		//! Set the minimum and maximum density to be shown in the canvas.
		void SetDensityRange
		(
			DynamicLib::Density,
			DynamicLib::Density
		);

		void SetFrequencyRange
		(
			DynamicLib::Rate,
			DynamicLib::Rate
		);

		void SetTimeRange
		(
			DynamicLib::Time,
			DynamicLib::Time
		);

		void SetPotentialRange
		(
			DynamicLib::Potential,
			DynamicLib::Potential
		);

	};

	class AsciiReportHandler : public AbstractReportHandler
	{
	public:

		AsciiReportHandler
		(
			const std::string&
		);

		AsciiReportHandler
		(
			const AsciiReportHandler&
		);

		virtual ~AsciiReportHandler
		(
		);

	};

	class SimulationRunParameter
	{
	public:

		//! standard constructor
		SimulationRunParameter
		( 
			const AbstractReportHandler&, 	
			unsigned int,					
  			DynamicLib::Time,   		
			DynamicLib::Time,
			DynamicLib::Time,		
			DynamicLib::Time,			
			DynamicLib::Time,
			const std::string&,				
			DynamicLib::Time report_state_time = 0		
		); 

		//! copy constructor
		SimulationRunParameter
		(
			const SimulationRunParameter&
		);

	};

	struct WilsonCowanParameter 
	{

		WilsonCowanParameter();


		//! constructor for convenience
		WilsonCowanParameter
		(
			//! membrane time constant in ms
			DynamicLib::Time,

			//! maximum firing rate in Hz
			DynamicLib::Rate,

			//! noise parameter for sigmoid
			double f_noise,
		
			//! input
			double f_input = 0
				
		);	

		virtual ~WilsonCowanParameter();

	};


	template <class Weight>
	class AbstractAlgorithm
	{
	public:

		AbstractAlgorithm(Number);

		virtual ~AbstractAlgorithm() = 0;
		

	}; // end of AbstractEvolutionAlgorithm

%template(D_AbstractAlgorithm) AbstractAlgorithm<double>;

	template <class WeightValue>
	class RateAlgorithm : public AbstractAlgorithm<WeightValue>
	{
	public:

		RateAlgorithm( DynamicLib::Rate );

		//! construct an algorithm that produces a stationary rate, whose values is maintained by an external variable
		RateAlgorithm( DynamicLib::Rate* );

		//! destructor
		virtual ~RateAlgorithm();


	}; // end of RateAlgorithm 

%template (D_RateAlgorithm) RateAlgorithm<double>;

	class  WilsonCowanAlgorithm : public AbstractAlgorithm<double>
	{
	public:

		WilsonCowanAlgorithm(const WilsonCowanParameter&);

		WilsonCowanAlgorithm(const WilsonCowanAlgorithm&);

		~WilsonCowanAlgorithm();



	}; // end of WilsonCowanAlgorithm



	//! This gives a DynamicNode's type, which will be checked when Dale's law is set.
	enum NodeType {NEUTRAL, EXCITATORY, INHIBITORY, EXCITATORY_BURST, INHIBITORY_BURST};

	struct SpatialPosition {

		SpatialPosition();

		SpatialPosition
		(
			float x,
			float y,
			float z,
			float f = 0
		);
	};

	template <class Weight>
	class DynamicNode //: public AbstractSparseNode<double,Weight>
	{
	public:

		DynamicNode();

		//! A DynamicNode receives its own clone from an AbstractAlgorithm instance,
		//! but an indivdual set of parameters
		DynamicNode
		(
			const AbstractAlgorithm<Weight>&,     
			NodeType type			       
		);

		DynamicNode
		(
			const DynamicNode<Weight>&
		);

   
		virtual ~DynamicNode();

		NodeType Type() const;

  
		void SetNodeName(const std::string&);

		//! Get a node's name
		std::string GetNodeName() const;

		//! Associate a spatial position with a node
		void AssociatePosition(const SpatialPosition&);

		//! Get the SpatialPosition associated with thes Node, if it exists.
		//! Function returns true if the position exists and false if not. The resulting position is then undefined
		bool GetPosition(SpatialPosition*) const;


	};

%template(D_DynamicNode) DynamicNode<double>;


	//! 
	template <class WeightValue>
	class DynamicNetworkImplementation //: 
	//	public SparseImplementation< DynamicNode<WeightValue> >
	{
	public:

		typedef WeightValue WeightType_;
		//! An empty implementation
		DynamicNetworkImplementation();

		//! make the node, labeled by the first nodeId, input other second Node
		bool MakeFirstInputOfSecond
			(
				NetLib::NodeId,
				NetLib::NodeId,
				const WeightValue&
			);


	}; // end of DynamicNetworkImplementation

%template(D_DynamicNetworkImplementation) DynamicNetworkImplementation<double>;


	template <class Implementation>
	class DynamicNetwork //: public Streamable
	{
	public:

	
		typedef typename Implementation::WeightType_ WeightType;

		DynamicNetwork
		(
		);


		~DynamicNetwork();

		NetLib::NodeId AddNode
			(
				const AbstractAlgorithm<WeightType>&,
				DynamicLib::NodeType
			);
		
		bool MakeFirstInputOfSecond
			(
				NetLib::NodeId,
				NetLib::NodeId,
				const WeightType&
			);




		bool ConfigureSimulation
			(
				const SimulationRunParameter&
			);

		bool Evolve();

		bool IsDalesLawSet() const;

		bool SetDalesLaw(bool);

		UtilLib::Number NumberOfNodes() const;


	}; // end of DynamicNetwork




} // end of DynamicLib

namespace PopulistLib {

	typedef double Efficacy;
	typedef double Potential;

	struct OrnsteinUhlenbeckConnection {

		//! In PopulistLib the connections have an associated inner product
//		typedef MuSigmaScalarProduct InnerProduct;

		//! effective number, may be fractional
		double		_number_of_connections;

		//! effective synaptic efficacy from one population on another
		Efficacy	_efficacy;

		//! default constructor
		OrnsteinUhlenbeckConnection
		(
		);

		//! construct, using effective number of connections and effectivie efficacy
		OrnsteinUhlenbeckConnection
		(
			double		number_of_connections,	//!< effective number of connections
			Efficacy	efficacy				//!< synaptic efficacy
		);
	};

	typedef OrnsteinUhlenbeckConnection PopulationConnection;
	typedef OrnsteinUhlenbeckConnection OUConnection;


	//! Parameters necessary for the configuration of an OUAlgorithm
	//!
	//! These are the parameters that define a leaky-integrate-and-fire neuron.

	struct OrnsteinUhlenbeckParameter {

		Potential _theta;				//!< threshold potential in V
		Potential _V_reset;				//!< reset potential in V
		Potential _V_reversal;			//!< reversal potential in V
	        DynamicLib::Time      _tau_refractive;		//!< (absolute) refractive time in s
	        DynamicLib::Time      _tau;					//!< membrane time constant in s

		//! default constructor
		OrnsteinUhlenbeckParameter();

		//! standard constructor
		OrnsteinUhlenbeckParameter
			(
				Potential theta,
				Potential V_reset,
				Potential V_reversal,
				DynamicLib::Time      tau_refractive,
				DynamicLib::Time      tau
			);

	}; // end of OrnsteinUhlenbeckParameter



	typedef OrnsteinUhlenbeckParameter PopulationParameter;

	struct InitialDensityParameter
	{
		//! default constructor
		InitialDensityParameter
		(
		):
		_mu(0),
		_sigma(0)
		{
		}

		//! constructor
		InitialDensityParameter
		(
			Potential mu,
			Potential sigma
		);

		//! copy constructor
		InitialDensityParameter
		(
			const InitialDensityParameter& rhs
		);

	};


	//! AbstractRebinner: Abstract base class for rebinning algorithms.
	//! 
	//! Rebinning algorithms serve to represent the density grid in the original grid, which is smaller
	//! than the current grid, because grids are expanding over time. Various ways of rebinning are conceivable
	//! and it may be necessary to compare different rebinning algorithms in the same program. The main simulation
	//! step in PopulationGridController only needs to know that there is a rebinning algorithm.
	class AbstractRebinner
	{
	public:

		//!
		virtual ~AbstractRebinner() = 0;


	};

	//! Interprolation rebinner is an important rebinning object
	//!
	//! Rebinning is necessary because the normal evolution of the population density takes
	//! place in an expanding array, whose growth may not exceed a maxmimum. Once the maximum
	//! is reached, rebinning must take place to fit the density profile in its original size.
	//! This rebinner first smooths the density profile around the reset bin, it then interpolates
	//! the points in the new density profile using cubic spline interpolation on the old density profile.
	//! Finally, the probability density that was taken from the reset bin is reintroduced at its new position.
	class InterpolationRebinner : public AbstractRebinner
	{
	public:

		//! Allocates spline resources in GLS
		InterpolationRebinner();

		//! destructor
		virtual ~InterpolationRebinner();


	};


	//! AbstractRateComputation
	//! There are several methods to calculate a Population's firing rate from the population density
	//! A virtual base class is provide, so that the methods can be exchanged in run time and the different
	//! methods can be compared within a single simulation

	class AbstractRateComputation {
	public:

		//! constructor
		AbstractRateComputation();

		virtual ~AbstractRateComputation() = 0;

	};

	//! IntegralRateComputation
	//! Computes the firing rate of a population from the density profile, using an integral method:
	//! \nu = \int^ \rho(v) dv
	class IntegralRateComputation : public AbstractRateComputation {
	public:

		//! constructor
		IntegralRateComputation();

	};


	//! A base class for all non-circulant solvers. 
	//! 
	//! The idea is that they can be exchanged during run time to investigate changes in the algorithm
	//! on the network level.
	class AbstractCirculantSolver {
	public:

		AbstractCirculantSolver();

		virtual ~AbstractCirculantSolver() = 0;

	};

	//! A base class for all non-circulant solvers. 
	//! 
	//! The idea is that they can be exchanged during run time to investigate changes in the algorithm
	//! on the network level.
	class AbstractNonCirculantSolver {
	public:

		AbstractNonCirculantSolver();

		//! virtual destructor for correct removal of allocated resources
		virtual ~AbstractNonCirculantSolver() = 0;

	};

	//! v_min and the expansion factor
	class PopulistSpecificParameter {

	public:

		//! default constructor
		PopulistSpecificParameter();

		//! copy constructor
		PopulistSpecificParameter
		(
			const PopulistSpecificParameter&
		);

		//! constructor
		PopulistSpecificParameter
		(
			Potential,								//!< minimum potential of the grid, (typically negative or below the reversal potential
			Number,									//!< initial number of bins
			Number,									//!< number of bins that is added after one zero-leak evaluation
			const InitialDensityParameter&,			//!< gaussian (or delta-peak) initial density profile
			double,									//!< expansion factor
			const string&	zeroleakequation_name	= "LIFZeroLeakEquations",	//!< The algorithm for solving the zero leak equations (see documentation at \ref AbstractZeroLeakequations if you want to modify the default choice)
			const string&	circulant_solver_name	= "CirculantSolver",		//!< The algorithm for solving the circulant equations (see documentation at \ref AbstractCirculant if you want to use a modofied version of this algorithm)
			const string&	noncirculant_solver_name= "NonCirculantSolver",		//!< The algorithm for solving the circulant equations (see documentation at \ref AbstractCirculant if you want to use a modofied version of this algorithm)
			const AbstractRebinner*	          = 0,  //!< Use when investigating alternatives to the standard rebinner, which InterpolationRebinner
			const AbstractRateComputation*    = 0  //!< Use when investigating alternatives to the standard rate computation, which is IntegralRateComputation    
		);
	};

	struct PopulistParameter {

		//! constructor
		PopulistParameter
		(
			const PopulationParameter&,
			const PopulistSpecificParameter&
		);
	};

} // end of namesspace PopulistLib

%template(OUAbstractAlgorithm) 	DynamicLib::AbstractAlgorithm<PopulistLib::OrnsteinUhlenbeckConnection>;

namespace PopulistLib {

	//! PopulationAlgorithm implements the simulation algorithm for leaky-integrate-and-fire neurons
	//! as developed by de Kamps (2003,2006,2008).
	//!
	//! PopulationAlgorithm simulates the reponse of an infinitly large population of spiking neurons
	//! in a response to a GAussian white noise input. The algorithm works in a DynamicNetwork which
	//! consists of DynamicNodes which are connected by binary valued connections, which are determined
	//! by an effective number of connections and an average efficacy. Examples of models which are
	//! based on this paradigm are Amit & Brunel (1997a,1997b), Brunel (1999). The idea is that each
	//! neuron in a population receives a large number of input spikes, each which contribute only a 
	//! small jump in the postsynaptic membrane potential. Under such assumptions, it can be shown that
	//! the membrane potential distribution is govenered by Fokker-Planck equations.
	//! This particular algorithm emulated such equations by choosing step sizes and input rates, which
	//! are not dictated by experimental value for rates and efficacies, but which emulate the mean and
	//! variance of the white noise. For a further discussion on the assumptions for the network and
	//! the conditions in which they hold see http://miind.sf.net

	class PopulationAlgorithm : public DynamicLib::AbstractAlgorithm<OrnsteinUhlenbeckConnection> {
	public:

		typedef PopulistParameter Parameter;

		//! Create a PopulistAlgorithm with settings defined in a PopulistParameter
		PopulationAlgorithm
		(
			const PopulistParameter&
		);

		//! copy constructor
		PopulationAlgorithm
		(
			const PopulationAlgorithm&
		);

		//! virtual destructor
		virtual ~PopulationAlgorithm();



	}; // end of PopulationAlgorithm

	//! ResponseParameterBrunel
	//! parameter as in Amit & Brunel (1997)
	struct ResponseParameterBrunel {

		double mu;
		double sigma;
		double theta;
		double V_reset;
		double tau_refractive;
		double tau;
	};

	// parameter as in Amit and Tsodyks (1991)

	struct ResponseParameterAmit {

		float I;
		float sigma;
		float tau;
		float tau_refractive;
		float theta;
		float mu;
	};

	double ResponseFunction( const PopulistLib::ResponseParameterAmit& );
	double ResponseFunction( const PopulistLib::ResponseParameterBrunel& );

} // end of PopulistLib
%template(D_DynamicNetwork)     DynamicLib::DynamicNetwork<DynamicLib::DynamicNetworkImplementation<double> >;
%template(OURateAlgorithm)     	DynamicLib::RateAlgorithm<PopulistLib::OrnsteinUhlenbeckConnection>;
// the instantiation of DynamicNetworkImplementation is crucial, since
// the type of the weight of a DynamicNetwork is obtained from its template
// argument, which is an implementation. If the DynamicNetworkImplementation is
// not instantiated, the DynamicNetwork can not determine which weight type
// is used by the Algorithms that populate the node.
%template(Pop_Imp)		DynamicLib::DynamicNetworkImplementation<PopulistLib::OrnsteinUhlenbeckConnection>;
%template(Pop_Net)		DynamicLib::DynamicNetwork<DynamicLib::DynamicNetworkImplementation<PopulistLib::OrnsteinUhlenbeckConnection> >;
