#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <TwoDLib.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace TwoDLib;

BOOST_AUTO_TEST_CASE(MeshTreeSimple){
	Mesh mesh("aexpoverview.mesh");
	MeshTree tree(mesh);

	Point p = tree.FindNearest(Point(-55.0,20.0));
	double d_large = 1e10;
	int i_d = 0;
	int j_d = 0;
	int k = 0;
	for (int i = 0; i < mesh.NrQuadrilateralStrips(); i++)
		for (int j = 0; j < mesh.NrCellsInStrip(i); j++){
			vector<Point> vec_p = mesh.Quad(i,j).Points();
			for (auto it = vec_p.begin(); it != vec_p.end(); it++){
				double d = ( (*it)[0] - p[0])*( (*it)[0] - p[0]) + ((*it)[1] - p[1])*((*it)[1] - p[1]);
				if (d < d_large){
					d_large = d;
					i_d = i;
					j_d = j;
					k = it - vec_p.begin();
				}
			}
		}

	BOOST_REQUIRE( mesh.Quad(i_d,j_d).Points()[k][0] == p[0] );
	BOOST_REQUIRE( mesh.Quad(i_d,j_d).Points()[k][1] == p[1] );
	std::cout << p[0] << std::endl;
	std::cout << p[1] << std::endl;
}


BOOST_AUTO_TEST_CASE(MeshNearest)
{
	Mesh mesh("aexpoverview.mesh");
	std::cout << "Mesh created" << std::endl;
	MeshTree tree(mesh);

	// uncomment this to see the  kdtree code at work; we don't use it in the project atm
//	vector<Point> vec_p = tree.FindNearestN(Point(-55.0,20.0),5.0);
//	for (auto it = vec_p.begin(); it != vec_p.end(); it++)
//		std::cout << (*it)[0] << " " << (*it)[1] << std::endl;

}

BOOST_AUTO_TEST_CASE(KdSimple){
#define rand1() (rand() / (double)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }

    int i;
    kd_node_t wp[] = {
        {{2, 3}}, {{5, 4}}, {{9, 6}}, {{4, 7}}, {{8, 1}}, {{7, 2}}
    };
    int len = sizeof(wp) / sizeof(wp[1]);
    struct kd_node_t testNode = {{9, 2}};
    struct kd_node_t *root, *found, *million;
    double best_dist;

    char buff[len*sizeof(kd_node_t)*sizeof(char)];
    root = make_tree(buff, wp, len, 0, 2);
    const int N = 1000000;
    int visited = 0;
    found = 0;
    nearest(root, &testNode, 0, 2, &found, &best_dist, &visited);

    // this is demo code and can be commented out to get a demonstration of the kdtree
//    printf(">> WP tree\nsearching for (%g, %g)\n"
//           "found (%g, %g) dist %g\nseen %d nodes\n\n",
//           testNode.x[0], testNode.x[1],
//            found->x[0], found->x[1], sqrt(best_dist), visited);

    million =(struct kd_node_t*) calloc(N, sizeof(struct kd_node_t));
    srand(time(0));
    for (i = 0; i < N; i++) rand_pt(million[i]);

    char* large_buffer = new  char[N*sizeof(kd_node_t)];
    root = make_tree(large_buffer, million, N, 0, 3);
    rand_pt(testNode);

    visited = 0;
    found = 0;
    nearest(root, &testNode, 0, 3, &found, &best_dist,&visited);

//    printf(">> Million tree\nsearching for (%g, %g, %g)\n"
//            "found (%g, %g, %g) dist %g\nseen %d nodes\n",
//            testNode.x[0], testNode.x[1], testNode.x[2],
//            found->x[0], found->x[1], found->x[2],
//            sqrt(best_dist), visited);

    /* search many random points in million tree to see average behavior.
       tree size vs avg nodes visited:
       10      ~  7
       100     ~ 16.5
       1000        ~ 25.5
       10000       ~ 32.8
       100000      ~ 38.3
       1000000     ~ 42.6
       10000000    ~ 46.7              */
    int sum = 0, test_runs = 100000;
    for (i = 0; i < test_runs; i++) {
        found = 0;
        visited = 0;
        rand_pt(testNode);
        nearest(root, &testNode, 0, 3, &found, &best_dist,&visited);
        sum += visited;
    }
//    printf("\n>> Million tree\n"
//            "visited %d nodes for %d random findings (%f per lookup)\n",
//            sum, test_runs, sum/(double)test_runs);

    free(million);
    free(large_buffer);
}


