%module Connectionism
%include "std_string.i" 
%include "std_vector.i"
%include "std_iostream.i" // this allows for istream constructor arguments
%{
	#include <fstream> // is necessary to pull off the istream trick
	#include <sstream>
	#include <vector>
	#include "../UtilLib/BasicDefinitions.h"
	#include "../NetLib/LayeredArchitecture.h"
	#include "../NetLib/PatternCode.h"
	#include "../NetLib/SigmoidParameter.h"
	#include "../SparseImplementationLib/LayeredSparseImplementation.h"
	#include "../SparseImplementationLib/LayerWeightIteratorThreshold.h"
	#include "../SparseImplementationLib/ReversibleSparseNodeCode.h"
	#include "../SparseImplementationLib/SparseImplementationCode.h"
        #include "../SparseImplementationLib/SparseImplementationTest.h"
	#include "../ConnectionismLib/BackpropTrainingVectorCode.h"
	#include "../ConnectionismLib/ConnectionismTest.h"
	#include "../ConnectionismLib/LayeredNetwork.h"
	#include "../ConnectionismLib/LayeredNetworkCode.h"
	#include "../ConnectionismLib/TrainingParameter.h"
	#include "../ConnectionismLib/TrainingUnit.h"
%}

// The following is necessary to make python "understand" that an ifstream is an istream
namespace std {
 
//	  class istream{}; not necessary

	  class ifstream : public istream{
	  public:
   	  ifstream(const char *fn);
   	  ~ifstream();
	  };
}

namespace UtilLib {

	  typedef unsigned int Number;
	  typedef unsigned int Index;
}

namespace NetLib {


  //! Pattern

  template <class PatternValue>
    class Pattern  {
    public:

    //! ctor
    Pattern();

    //! ctor, predefined size
    Pattern( UtilLib::Number );

    //! copy ctor
    Pattern( const Pattern& );

    //!  dtor
    virtual ~Pattern();

    // element access:
    %extend { 
    PatternValue __getitem__(int i) const {
    		 	return self->operator[](i);
		 }
    void __setitem__(int i, PatternValue d){
    	 		self->operator[](i)=d;
    	 	 }
    	 
    };

    //! Pattern length
    UtilLib::Number Size() const;

    //! Set all elements to 0
    void Clear();


  };


	  
	  
  struct NodeId 
  {
          //! Default constructor
	  explicit NodeId():_id_value(0){}

	  //! Explict construction
	  explicit NodeId(int id_value ):_id_value(id_value){}
	  
	  //! destructor
	  ~NodeId(){}

}; // end of NodeId


	//! SigmoidParameter
	class SigmoidParameter
	{
	public:

		//! default constructor, intialization to zero
		SigmoidParameter();

		//! explicit constructor
		SigmoidParameter
			(
				double f_min_act, 
				double f_max_act, 
				double f_noise
			):_f_min_act(f_min_act),
			  _f_max_act(f_max_act),
			  _f_noise(f_noise)
		{
		}

	}; // end of SigmoidParameter

	  class LayeredArchitecture {
	  public:

		LayeredArchitecture(const std::vector<unsigned int>&, bool b_threshold = false);

	       virtual ~LayeredArchitecture();

	       UtilLib::Number NumberOfLayers() const;
	       
	       virtual unsigned int NumberOfInputNodes() const;
	       
	       virtual unsigned int NumberOfOutputNodes() const;
	  };
}

namespace SparseImplementationLib {


	template <class NodeType>
	class LayerWeightIteratorThreshold
	{
	public:

		LayerWeightIteratorThreshold();

	};

	  class SparseImplementationTest {
	  public:
	  
		SparseImplementationTest();

		NetLib::LayeredArchitecture
			CreateXORArchitecture  () const;

	  };

	template <class ActivityType_, class WeightType_>
	class AbstractSparseNode 
	{
	public:


		typedef ActivityType_	ActivityType;
		typedef WeightType_		WeightType;

		//! Standard constructor, sets a NodeId
		AbstractSparseNode(NetLib::NodeId);

		//! copy constructor
		AbstractSparseNode(const AbstractSparseNode<ActivityType, WeightType>& );

		//! virtual destructor
		virtual ~AbstractSparseNode() = 0;
	};

	template <class NodeValue, class WeightValue>
	class ReversibleSparseNode : public AbstractSparseNode<NodeValue,WeightValue>{
	
	public:

		ReversibleSparseNode();

	}; // end of ReversibleSparseNode


	template <class NodeType_>
	class SparseImplementation{
	public:

		//! allows to deduce NodeType from implementation template argument
		typedef NodeType_ NodeType;
		
		//! allows to deduce type of Nodes activation value and of weights
		typedef typename NodeType_::ActivityType NodeValue;
		typedef typename NodeType_::WeightType   WeightValue;

		//! Empty implementation
		SparseImplementation();

        };


  template <class NodeType>
  class LayeredSparseImplementation : public SparseImplementation<NodeType> {

  public:

    typedef LayerWeightIterator<NodeType>           WeightLayerIterator;
    typedef LayerWeightIteratorThreshold<NodeType>  WeightLayerIteratorThreshold;

    //! Construct from stream (file) 
    LayeredSparseImplementation(istream&);

    //! Construct from LayeredArchitecture
    LayeredSparseImplementation(LayeredArchitecture*);

    //!
    LayeredSparseImplementation(const LayeredSparseImplementation&);


    //!
    virtual ~LayeredSparseImplementation();
 


  }; // end of LayeredSparseImplementation


} // end of SparseImplementationLib





namespace ConnectionismLib {

    template <class _Implementation>
    class LayeredNetwork 
    {
    public:


		typedef typename _Implementation::NodeType NodeType;

		// ctors, dtors and the like:
				
		LayeredNetwork
		(
			std::istream&
		);

		//! Create from a LayeredArchitecture
		LayeredNetwork
		(
		        NetLib::LayeredArchitecture*
		 );

		//! Use default squashing function (Sigmoid), but override default sigmoid parameter
		LayeredNetwork
		(
		        NetLib::LayeredArchitecture*, 
			const NetLib::SigmoidParameter&
		);


%extend {
		LayeredNetwork
		(
		        const std::string& str
		){
			std::ifstream stream(str.c_str());
			return new ConnectionismLib::LayeredNetwork<_Implementation >(stream);
		}
};

		//! Enter pattern in Input Nodes
		 bool ReadIn(const NetLib::Pattern<typename _Implementation::NodeValue>&);

		//! Read pattern from output Nodes
		NetLib::Pattern<typename _Implementation::NodeValue> ReadOut() const;


		//! Evolve the Network in an Order, prescribed by Order 
		virtual bool  Evolve();


		UtilLib::Number NumberOfInputNodes () const;
		UtilLib::Number NumberOfOutputNodes() const;
		UtilLib::Number NumberOfNodes      () const;

%extend {

		//TODO: cludge which creates memory leaks; introduce a member variable for python cases which will be cleaned upon destruction
		char* __str__() { std::ostringstream s; $self->ToStream(s); string str = s.str(); char* p = new char[str.size()+1]; strcpy(p,str.c_str()); return p;}
}

    }; // end of LayeredNetwork

	//! TrainingUnit

	template <class PatternValue>
	class TrainingUnit
	{
	public:

		TrainingUnit();
		TrainingUnit(const TrainingUnit&);
		TrainingUnit
		(
			const NetLib::Pattern<PatternValue>&, 
			const NetLib::Pattern<PatternValue>&
		);
			

		const NetLib::Pattern<PatternValue>& InPat()  const;
		const NetLib::Pattern<PatternValue>& OutPat() const;

	};



	//! TrainingParameter

	struct TrainingParameter 
	{

		double _f_stepsize;
		double _f_sigma;
		double _f_bias;
		size_t _n_step;
		bool   _train_threshold;
		double _f_threshold_default;
		double _f_momentum;
		long   _l_seed;
		size_t _n_init;

		TrainingParameter();
		TrainingParameter
		( 
			double f_stepsize,
			double f_sigma,
			double f_bias,
			size_t n_step,
			bool   b_train_threshold,
			double f_threshold_default,
			double f_momentum,
			long   l_seed,
			size_t n_init 
		);

	}; // end of TrainingParameter

	enum TrainingMode {BATCH, RANDOM};

	//! BackpropTrainingVector
	template <class LayeredImplementation, class Iterator>
	class BackpropTrainingVector {
   	 public:

    	 BackpropTrainingVector
    	 ( 
		LayeredNetwork<LayeredImplementation>* ,  
		const TrainingParameter&, 
		TrainingMode = ConnectionismLib::RANDOM 
	);
 
	
	//! virtual destructor
	virtual ~BackpropTrainingVector();



    	bool PushBack                
	(
		const TrainingUnit<typename LayeredImplementation::NodeValue>&
	);


    	size_t Size() const;

	// training related methods:
	void	Train();
	
	double	ErrorValue() const;
 	};


	class ConnectionismTest 
	{
	public:
    
		ConnectionismTest();

		bool Execute();


	};
 
} // end of ConnectionismLib




%template (I_Vector) std::vector<unsigned integer>;
%template (D_Pattern) NetLib::Pattern<double>;
%template (D_AbstractSparseNode) SparseImplementationLib::AbstractSparseNode<double,double>;
%template (D_ReversibleSparseNode) SparseImplementationLib::ReversibleSparseNode<double,double>;
%template (D_ReversibleSparseImplementation) SparseImplementationLib::SparseImplementation<SparseImplementationLib::ReversibleSparseNode<double,double> >;
%template (D_LayeredNetworkImplementation) SparseImplementationLib::LayeredSparseImplementation<SparseImplementationLib::ReversibleSparseNode<double,double> >;
%template (D_LayeredNetwork) ConnectionismLib::LayeredNetwork< SparseImplementationLib::LayeredSparseImplementation<SparseImplementationLib::ReversibleSparseNode<double,double> > >;
%template (D_TrainingUnit) ConnectionismLib::TrainingUnit<double>;
%template (D_LayerWeightIteratorThreshold) SparseImplementationLib::LayerWeightIteratorThreshold<SparseImplementationLib::ReversibleSparseNode<double,double> >;
%template (D_BackpropAlgorithm) ConnectionismLib::BackpropTrainingVector<SparseImplementationLib::LayeredSparseImplementation<SparseImplementationLib::ReversibleSparseNode<double,double> >, SparseImplementationLib::LayeredSparseImplementation<SparseImplementationLib::ReversibleSparseNode<double,double> >::WeightLayerIteratorThreshold  >;
