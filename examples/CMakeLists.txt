
set( AEXP
aexp/aexp_-1_0_0.mat
aexp/aexp_1_0_0.mat
aexp/aexpanalysis.py
aexp/aexp.model
aexp/aexpnoa642fee6a-9257-4624-9178-9a404b8bad50_-1_0_0_0_.mat
aexp/aexpnoa642fee6a-9257-4624-9178-9a404b8bad50_1_0_0_0_.mat
aexp/aexpnoa642fee6a-9257-4624-9178-9a404b8bad50.model
aexp/aexp.xml
)

set( COND
cond/cond_0_0.005_0.mat
cond/condanalysis.py
cond/cond.xml
cond/cond_0_0.05_0.mat
cond/cond.model
cond/mc_gain_curves.py
)

set( LIF
lif/lif_0.01_0_0_0_.mat
lif/lif.model
lif/lifanalysis.py
lif/lif.xml
lif/steady_state.dat
)

set( LIF_PYTHON
lif_python/lif.fid
lif_python/lif.model
lif_python/lif.rev
lif_python/lif.xml
lif_python/lif_0.01_0_0_0_.mat
lif_python/lif.mesh
lif_python/lif.res
lif_python/lif.stat
lif_python/runlif.py
)

set( QIF
qif/qif_0.005_0_0.mat
qif/qifanalysis.py
qif/qif_inhib_0.05_0_0.mat
qif/qif_0.03_0_0.mat
qif/qif_excit_-0.05_0_0.mat
qif/qif_inhib.model
qif/qif_-0.05_0_0.mat
qif/qif_excit_0.05_0_0.mat
qif/qif.model
qif/qif_0.05_0_0.mat
qif/qif_excit.model
qif/qif.xml
qif/qif_0.15_0_0.mat
qif/qif_inhib_-0.05_0_0.mat
qif/show_qif_gain.py
)

set( RATE
rate_based/analyse_rate.py
rate_based/analytic.dat
rate_based/rate.xml
rate_based/response_curve.py
)

set( RESP
response_curve_lif/analyse_response.py
response_curve_lif/response_analyse.py
response_curve_lif/response_curve.py
response_curve_lif/response.xml
)

set( SINGLE
single_lif/analyse_single_lif.py
single_lif/no_decay.xml
single_lif/no_input.xml
single_lif/response.xml
single_lif/single_lif.py
)

set( SELF
single_selfinhib/self.xml
)

set( TWO
twopop/twopop.xml
)

set( GRID_COND
grid_examples/cond/cond.model
grid_examples/cond/cond.xml
grid_examples/cond/cond_0_0_0_0_.tmat
)

set( GRID_FN
grid_examples/fn/fn.model
grid_examples/fn/fn.xml
grid_examples/fn/fn_0_0_0_0_.tmat
)

install( FILES ${AEXP}
    DESTINATION share/miind/examples/aexp )
install( FILES ${COND}
    DESTINATION share/miind/examples/cond )
install( FILES ${LIF}
    DESTINATION share/miind/examples/lif )
install( FILES ${LIF_PYTHON}
    DESTINATION share/miind/examples/lif_python )
install( FILES ${QIF}
    DESTINATION share/miind/examples/qif )
install( FILES ${RATE}
    DESTINATION share/miind/examples/rate_based )
install( FILES ${RESP}
    DESTINATION share/miind/examples/response_curve_lif )
install( FILES ${SINGLE}
    DESTINATION share/miind/examples/single_lif )
install( FILES ${SELF}
    DESTINATION share/miind/examples/single_selfinhib )
install( FILES ${TWO}
    DESTINATION share/miind/examples/twopop )
install( FILES ${GRID_COND}
    DESTINATION share/miind/examples/grid_examples/cond )
install( FILES ${GRID_FN}
    DESTINATION share/miind/examples/grid_examples/fn )
