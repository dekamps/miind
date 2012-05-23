#! /usr/bin/python

import ExperimentSet

if __name__ == '__main__':
	ExperimentSet.do_it( [ ( "horizontal_bar", ( ( ( ( 10, 10 ), ( 80, 80 ) ), ( 3, "white", 255 ) ) ) ), ( "vertical_bar", ( ( ( 80, 80 ), ( 80, 80 ) ), ( 3, "white", 255 ) ) ) ], size = ( 160, 160 ) )
