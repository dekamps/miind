/**
\page changes How to adapt the old code to the new MPI Library

\section overview General Changes
\subsection error Error Handling
The error handling was switched to exceptions. Therefore several methods that previously returned
bool values to show their state are now void methods.
\subsection log Logging
The logging was changed significantly. For details see \ref logging.
The main change is now that you can define the level of your log or debug message.
Messages with a finer level are removed during the compilation such that no overhead occur.
\subsection gra Graphical Representation
As it makes no sense to watch the simulation at runtime on a cluster all features regarding the graphical representation
were removed.
\subsection gen Generation of Nodes
The generation of nodes is encapsulated from the user. Therefore do not assume anything about the numbers given
to the nodes. The only valid assumption is that the first node gets the number 0 instead if 1 as previous.
\section Applications
Several method names should be adapted to the new interface of the MPINetwork.
The following methods are now named differently from the old ones.
<ul>
<li>MakeFirstInputOfSecond -> makeFirstInputOfSecond</li>
<li>AddNode -> addNode</li>
<li>Evolve -> evolve</li>
<li>ConfigureSimulation -> configureSimulation</li>
</ul>
These are the most important changes on the naming scheme. However other methods were also renamed or dropped.


\subsection makeFirstInputOfSecond
This method now does now additionally take a forth parameter which correspond to the NodeType of the second node.
\subsection addNode
The generation of the node is now encapsulated. Therefore the addNode method needs to be provided with an algorithm
and the Type of the node. The return value stays the same. However the node numbers are handled internally. You
should not assume anything on the numbers given to the nodes!
\subsection mpi The MPI environment
Never make direct calls to the mpi environment always use the proxy class MPILib::utilities::MPIProxy. Only in the main function
of you application you need to add the following lines:
\code
...
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#endif

...
int main(int argc, char* argv[]) {
#ifdef ENABLE_MPI
	// initialise the mpi environment this cannot be forwarded to a class
	boost::mpi::environment env(argc, argv);
#endif
...
all your code here
...
	} catch (std::exception& exc) {
		std::cout << exc.what() << std::endl;
#ifdef ENABLE_MPI
		//Abort the MPI environment in the correct way :)
		env.abort(1);
#endif
	}
\endcode
\section Algorithms
The major changes includes the switch from iterators to now passed vectors. Therefore the
method \c EvolveNodeState and \c CollectExternalInput need adaption. See below for details.


Several method names should be adapted to the new MPILib::algorithm::AlgorithmInterface of MPILib.
The following methods are now named differently from the old ones.
<ul>
<li>EvolveNodeState -> evolveNodeState</li>
<li>CollectExternalInput -> prepareEvolve</li>
<li>Configure -> configure</li>
<li>CurrentTime -> getCurrentTime</li>
<li>CurrentRate -> getCurrentRate</li>
<li>Grid -> getGrid</li>
</ul>
All other methods of the old AbstractAlgorithm were dropped.
\subsection evolveNodeState
This method was changed significantly. It now gets vectors were the actual values are stored passed instead
of iterators like in the old version.
This method can be overloaded depending on the needs of an algorithm. One method provides
A Vector of the rates of the precursors, a vector of the weights of the connection to the precursors
and the current time. Additional the if an algorithm also needs the types of the precursors the overloaded method
evolveNodeState should be used where additionally a vector of the types of the precursors is passed. However only
override one of the two provided methods. For more details see the documentation of AlgorithmInterface.
\subsection prepareEvolve
This method replaces the old CollectExternalInput method. This method also provides access to the data via passed
vectors and not iterators as before.

\subsection Example adaptions
For an example I use the implementation of the WilsonCowanAlgorithm
The old code looked like that (All not interesting code was removed for clarity):
\code
bool WilsonCowanAlgorithm::EvolveNodeState
(
	predecessor_iterator iter_begin,
	predecessor_iterator iter_end,
	Time time_to_achieve
)
{
	double f_inner_product = InnerProduct(iter_begin, iter_end);
	...
	return true;
}
\endcode
The code of the InnerProduct look like that:
\code
	double AbstractAlgorithm<WeightValue>::InnerProduct
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	) const
	{

		Connection* p_begin = iter_begin.ConnectionPointer();
		Connection* p_end	= iter_end.ConnectionPointer();

		return inner_product
			(
				p_begin,
				p_end,
				p_begin,
				0.0,
				plus<double>(),
				SparseImplementationLib::ConnectionProduct<double,WeightValue>()
			);
	}
\endcode

In the new WilsonCowanAlgorithm the reduced evolveNodeState look like that:
\code
void WilsonCowanAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	double f_inner_product = innerProduct(nodeVector, weightVector);
	...
}
\endcode
And reduced innerProduct like that:
\code
double WilsonCowanAlgorithm::innerProduct(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector) {
	...
	return std::inner_product(nodeVector.begin(), nodeVector.end(), weightVector.begin(), 0.0);
}
\endcode
The both code examples have exactly the same behavior.

*/
