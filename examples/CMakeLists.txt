
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
qif/qifanalysis.py
qif/qif.model
qif/qif.xml
qif/show_qif_gain.py
)

set( TVB
miind_tvb/tvb_miind_sim.py
miind_tvb/grid_izhikevich.py
miind_tvb/izh.model
miind_tvb/izh.xml
miind_tvb/izh_0_0_0_0_.tmat
)

set( CLI_PLOTS
cli_plots/example.xml
cli_plots/grid_izhikevich.py
cli_plots/izh.model
cli_plots/izh.projection
cli_plots/izh_0_0_0_0_.tmat
cli_plots/marginal_density.npz
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
single_lif/response.xml
single_lif/single_lif.py
)

set( SELF
single_selfinhib/self.xml
)

set( TWO
twopop/twopop.xml
)

set( MA_ADEX
model_archive/Adex/adex.xml
model_archive/Adex/adex.model
model_archive/Adex/adex_0_0_0_0_.tmat
)

set( MA_COND
model_archive/Conductance2D/cond.xml
model_archive/Conductance2D/cond.model
model_archive/Conductance2D/cond_0_0_0_0_.tmat
)

set( MA_EIF
model_archive/EIF/eif.xml
model_archive/EIF/eif.model
model_archive/EIF/eif_0.1_0_0_0_.mat
)

set( MA_EPILEPTOR
model_archive/Epileptor/epi.xml
model_archive/Epileptor/epileptor_ict.model
model_archive/Epileptor/epileptor_ict_0_0_0_0_.tmat
model_archive/Epileptor/epileptor_intict.model
model_archive/Epileptor/epileptor_intict_0_0_0_0_.tmat
)

set( MA_FN
model_archive/FitzhughNagumo/fn.xml
model_archive/FitzhughNagumo/fn.model
model_archive/FitzhughNagumo/fn_0_0_0_0_.tmat
)

set( MA_IZH
model_archive/Izhikevich/izh.xml
model_archive/Izhikevich/izh.model
model_archive/Izhikevich/izh_0_0_0_0_.tmat
)

set( MA_LIF
model_archive/LIF/lif.xml
model_archive/LIF/lif.model
model_archive/LIF/lif_0.01_0_0_0_.mat
)

set( MA_MN
model_archive/MinimalMotorneuron/mn.xml
model_archive/MinimalMotorneuron/mn_dendrite.model
model_archive/MinimalMotorneuron/mn_dendrite_0_0_0_0_.tmat
model_archive/MinimalMotorneuron/mn_soma.model
model_archive/MinimalMotorneuron/mn_soma_0_0_0_0_.tmat
)

set( MA_PNAP
model_archive/PersistentSodium2D/pnap.xml
model_archive/PersistentSodium2D/pnap.model
model_archive/PersistentSodium2D/pnap.tmat
)

set( MA_QIF
model_archive/QIF/qif.xml
model_archive/QIF/qif.model
model_archive/QIF/qif_0.01_0_0_0_.mat
)

set( MA_WC
model_archive/WilsonCowan/wc.xml
)

set( TUTORIAL
tutorial/Tutorial\ MIIND.pdf
)

set( TUTORIAL_GRID
tutorial/grid/izh.xml
tutorial/grid/izhikevich.py
)

set( TUTORIAL_T1
tutorial/t1/t1.mesh
tutorial/t1/t1.model
tutorial/t1/t1.projection
tutorial/t1/t1.xml
tutorial/t1/t1_1_0_0_0_.mat
)

set( TUTORIAL_T2
tutorial/t2/t2.mesh
tutorial/t2/t2.model
tutorial/t2/t2.projection
tutorial/t2/t2.xml
tutorial/t2/t2_0_0.05_0.mat
)

set( TUTORIAL_T3
tutorial/t3/t3.mesh
tutorial/t3/t3.model
tutorial/t3/t3.projection
tutorial/t3/t3.xml
tutorial/t3/t3_0.1_0_0_0_.mat
)

set( TUTORIAL_VECTOR
tutorial/vectorised/adex.model
tutorial/vectorised/adex_1_0_0_0_.mat
tutorial/vectorised/adex_-1_0_0_0_.mat
tutorial/vectorised/vector.xml
)

set( QUICK_START
quick_start/cond.xml
quick_start/generateCondFiles.py
)

ADD_SUBDIRECTORY(IntegrationTests)

set(EXAMPLES_BASE_DIR share/miind/examples)

install( FILES ${AEXP}
    DESTINATION ${EXAMPLES_BASE_DIR}/aexp )
install( FILES ${COND}
    DESTINATION ${EXAMPLES_BASE_DIR}/cond )
install( FILES ${LIF}
    DESTINATION ${EXAMPLES_BASE_DIR}/lif )
install( FILES ${LIF_PYTHON}
    DESTINATION ${EXAMPLES_BASE_DIR}/lif_python )
install( FILES ${QIF}
    DESTINATION ${EXAMPLES_BASE_DIR}/qif )
install( FILES ${TVB}
    DESTINATION ${EXAMPLES_BASE_DIR}/miind_tvb )
install( FILES ${QUICK_START}
    DESTINATION ${EXAMPLES_BASE_DIR}/quick_start )
install( FILES ${RATE}
    DESTINATION ${EXAMPLES_BASE_DIR}/rate_based )
install( FILES ${RESP}
    DESTINATION ${EXAMPLES_BASE_DIR}/response_curve_lif )
install( FILES ${SINGLE}
    DESTINATION ${EXAMPLES_BASE_DIR}/single_lif )
install( FILES ${SELF}
    DESTINATION ${EXAMPLES_BASE_DIR}/single_selfinhib )
install( FILES ${CLI_PLOTS}
    DESTINATION ${EXAMPLES_BASE_DIR}/cli_plots )
install( FILES ${TWO}
    DESTINATION ${EXAMPLES_BASE_DIR}/twopop )
install( FILES ${MA_ADEX}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/Adex )
install( FILES ${MA_COND}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/Conductance2D )
install( FILES ${MA_EIF}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/EIF )
install( FILES ${MA_EPILEPTOR}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/Epileptor )
install( FILES ${MA_FN}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/FitzhughNagumo )
install( FILES ${MA_IZH}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/Izhikevich )
install( FILES ${MA_LIF}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/LIF )
install( FILES ${MA_MN}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/MinimalMotorneuron )
install( FILES ${MA_PNAP}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/PersistentSodium2D )
install( FILES ${MA_QIF}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/QIF )
install( FILES ${MA_WC}
    DESTINATION ${EXAMPLES_BASE_DIR}/model_archive/WilsonCowan )
install( FILES ${TUTORIAL}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial )
install( FILES ${TUTORIAL_GRID}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial/grid )
install( FILES ${TUTORIAL_T1}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial/t1 )
install( FILES ${TUTORIAL_T2}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial/t2 )
install( FILES ${TUTORIAL_T3}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial/t3 )
install( FILES ${TUTORIAL_VECTOR}
    DESTINATION ${EXAMPLES_BASE_DIR}/tutorial/vectorised )
