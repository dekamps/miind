import xml.etree.ElementTree as ET


ALGORITHMS = { 'RateAlgorithm'   : {'Connection' : 'double', 'Parameter': '' },
               'RateAlgorithm'   : {'Connection' : 'DelayedConnection', 'Parameter' : ''},
               'OUAlgorithm'     : {'Connection' : 'DelayedConnection', 'Parameter' : 'NeuronParameter'},
               'GeomAlgorithmDC' : {'Connection' : 'DelayedConnection', 'Parameter' : 'GeomParameter'},
               'MeshAlgorithm'   : {'Connection' : 'DelayedConnection', 'Parameter': ''},
               'GridAlgorithm'   : {'Connection' : 'CustomConnectionParameters', 'Parameter': ''},
               'GridJumpAlgorithm'   : {'Connection' : 'CustomConnectionParameters', 'Parameter': ''} ,
               'GridSomaDendriteAlgorithm'   : {'Connection' : 'CustomConnectionParameters', 'Parameter': ''}}

ALGORITHM_NAMES = {}

RATEFUNCTIONS = []

def Register(name, cpp_name):
    if name in ALGORITHM_NAMES.keys():
        raise NameError('Name already exists; pick unique algorithm names.')
    else:
        ALGORITHM_NAMES[name] = cpp_name

def setup_test():
    f = open('bla.xml')
    sim = ET.fromstring(f.read())
    algorithms = sim.findall('Algorithms/Algorithm')
    weighttype = sim.find('WeightType')
    return algorithms, weighttype

def parse_rate_algorithm(alg, i , weighttype):
    s = ''
    if not 'name' in  alg.keys():
        raise NameError('Name tag expected')

    cpp_name = 'rate_alg_' + str(i)
    Register(alg.attrib['name'],cpp_name)

    s += '\tMPILib::RateAlgorithm<' + weighttype.text + '> ' + cpp_name +  '('
    rt = alg.find('rate')
    s += rt.text
    s += ');\n'

    return s

def parse_neuron_parameter(np,i):
    s='\tGeomLib::NeuronParameter par_neur_' + str(i) +'('
    v_thr=np.find('V_threshold')
    s += v_thr.text + ','
    v_res=np.find('V_reset')
    s += v_res.text + ','
    v_rev=np.find('V_reversal')
    s += v_rev.text + ','
    t_ref=np.find('t_refractive')
    s += t_ref.text + ','
    t_mem=np.find('t_membrane')
    s += t_mem.text + ');\n'

    return s

def parse_individual_parameters(alg,i):
    s=''

    v_min = alg.find('OdeParameter/V_min')
    s = '\tconst MPILib::Potential v_min_' + str(i) + ' = ' + v_min.text + ';\n'

    n_bins = alg.find('OdeParameter/N_bins')
    s += '\tconst MPILib::Number n_bins_' + str(i) + ' = ' + n_bins.text + ';\n'

    return s

def parse_initialdensity_parameter(idp,i):
    s='\tconst GeomLib::InitialDensityParameter par_dense_' + str(i) + '('
    mu=idp.find('mu')
    s += mu.text + ', '
    sigma = idp.find('sigma')
    s += sigma.text + ');\n'

    return s

def parse_diffusion_parameter(i,alg):
    s = ''
    dp=alg.find('DiffusionParameter')
    dl=dp.find('diffusion_limit')
    dj=dp.find('diffusion_jump')

    s += '\tGeomLib::DiffusionParameter par_diffusion_' + str(i) +'('
    s += dj.text + ','
    s += dl.text + ');\n'

    return s

def parse_current_compensation_parameter(i,alg):
    s = ''
    cc=alg.find('CurrentCompensationParameter')
    mu=cc.find('mu')
    si=cc.find('sigma')

    s += '\tGeomLib::CurrentCompensationParameter par_current_' + str(i) + '('
    s += mu.text + ','
    s += si.text + ');\n'

    return s

def DetermineSpikingSystem(i,alg):
    sodes = alg.find('GeomAlgorithm/SpikingNeuralDynamics')

    s = '\tGeomLib::'  + sodes.attrib['type']
    s += ' dyn_'     + str(i) + '('
    s += 'par_ode_'  + str(i) +','
    s += 'par_qif_'  + str(i) + ');\n'

    return s

def parse_current_parameter(i,alg):
    s = ''
    qp=alg.find('SpikingNeuralDynamics/QifParameter')
    ii=qp.find('I')
    s += '\tGeomLib::QifParameter par_qif_' + str(i) + '('
    s += ii.text + ',' + ii.text + ');\n'

    return s

def publish_spiking_ode_system(i,alg):
    s=''
    s += parse_current_parameter(i,alg)
    s += parse_diffusion_parameter(i,alg)
    s += parse_current_compensation_parameter(i,alg)
    s += DetermineSpikingSystem(i,alg)
    s += '\tGeomLib::QifOdeSystem sys_ode_' + str(i) + '('
    s += 'dyn_' + str(i) + ');\n'

    return s

def publish_leaking_ode_system(lambdavalue,i):
    s = ''

    s += '\tGeomLib::LifNeuralDynamics dyn_ode_leak_' + str(i) + '('
    s += 'par_ode_'     + str(i) + ', '
    s += lambdavalue    + ');\n'
    # ode sytem must be determined from the file
    s += '\tGeomLib::LeakingOdeSystem sys_ode_' + str(i) +'('
    s += 'dyn_ode_leak_' + str(i)  + ');\n'
    return s

def  NonDefaultGeomParameterArgs(system, i):
    if system == 'LeakingOdeSystem':
        return ',\"LifNumericalMasterEquation\"'
    if system == 'SpikingOdeSystem':
        return ',\"SpikingSystemMasterEquation\", par_diffusion_' + str(i) + ', par_current_' + str(i)

def wrap_up_geom_algorithm(alg, i):
    s = ''

    d=alg.attrib

    if not 'name' in  d.keys():
        raise NameError('name tag expected')

    s += '\tGeomLib::OdeParameter par_ode_' + str(i) + '('
    s += 'n_bins_'      + str(i) + ','
    s += 'v_min_'       + str(i) + ','
    s += 'par_neur_'    + str(i) + ','
    s += 'par_dense_'   + str(i) + ');\n'

    if d['system'] == 'LeakingOdeSystem':
        if not 'lambda' in d.keys():
            raise NameError('LeakingOdeSystem requires a lambda parameter')
        else:
            s += publish_leaking_ode_system(d['lambda'],i)
    else:
        if d['system'] == 'SpikingOdeSystem':
            s += publish_spiking_ode_system(i,alg)
        else:
            raise NameError('Unknown system type')


    # default parameters must be handled by GeomParameter
    s += '\tGeomLib::GeomParameter par_geom_' + str(i) + '('
    s += 'sys_ode_'+ str(i)
    s +=  NonDefaultGeomParameterArgs(d['system'],i)
    s += ');\n'
    cpp_name = 'alg_geom_' + str(i)
    s += '\tGeomLib::GeomAlgorithm<DelayedConnection> ' + cpp_name + '('
    s += 'par_geom_' + str(i)
    s += ');\n'
    s += '\n'
    Register(d['name'], cpp_name)

    return s

def parse_geom_algorithm(alg, i, weighttype):

    if alg.attrib['type'] != 'GeomAlgorithm':
        raise ValueError
    s = ''
    s += parse_individual_parameters(alg,i)

    np=alg.find('OdeParameter/NeuronParameter')
    str_np =  parse_neuron_parameter(np,i)
    s+= str_np

    idp = alg.find('OdeParameter/InitialDensityParameter')
    s += parse_initialdensity_parameter(idp,i)

    s += wrap_up_geom_algorithm(alg, i)

    return s

def parse_mesh_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'MeshAlgorithm':
        raise ValueError

    if weighttype.text ==  "DelayedConnection":
        type = "MPILib::" + weighttype.text
    else:
        type = "double"


    s += '\tstd::vector<std::string> '
    vec_name = 'vec_mat_' + str(i)
    s += vec_name + '{\"'

    matfilelist = alg.findall('MatrixFile')
    # don't use i below
    for k, fl in enumerate(matfilelist):
        if k > 0:
            s +='\",\"'
        s += fl.text
    s += '\"};\n'

    timestep = alg.find('TimeStep')

    cpp_name = 'alg_mesh_' + str(i)

    if 'solver' in alg.attrib.keys():
        # solver type must be provided without TwoDLib::
        s += '\tTwoDLib::MeshAlgorithm<'+type+',TwoDLib::' + alg.attrib['solver'] + '> ' + cpp_name + '(\"'
    else:

        s += '\tTwoDLib::MeshAlgorithm<'+type+'> ' + cpp_name + '(\"'
    s += alg.attrib['modelfile'] + '\",' + vec_name + ',' + timestep.text

    if 'tau_refractive' in alg.keys():
        s += ', '  + alg.attrib['tau_refractive']
    else:
        s += ', 0.0'
    if 'ratemethod' in alg.keys():
        s += ', '  + "\"" + alg.attrib['ratemethod'] + "\""
    s += ');\n'


    Register(alg.attrib['name'], cpp_name)

    return s

def parse_mesh_algorithm_custom(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'MeshAlgorithmCustom':
        raise ValueError

    if weighttype.text ==  "CustomConnectionParameters":
        type = "MPILib::" + weighttype.text
    else:
        type = "double"


    s += '\tstd::vector<std::string> '
    vec_name = 'vec_mat_' + str(i)
    s += vec_name + '{\"'

    matfilelist = alg.findall('MatrixFile')
    # don't use i below
    for k, fl in enumerate(matfilelist):
        if k > 0:
            s +='\",\"'
        s += fl.text
    s += '\"};\n'

    timestep = alg.find('TimeStep')

    cpp_name = 'alg_mesh_' + str(i)

    if 'solver' in alg.attrib.keys():
        # solver type must be provided without TwoDLib::
        s += '\tTwoDLib::MeshAlgorithmCustom<TwoDLib::' + alg.attrib['solver'] + '> ' + cpp_name + '(\"'
    else:

        s += '\tTwoDLib::MeshAlgorithmCustom<TwoDLib::MasterOdeint> ' + cpp_name + '(\"'
    s += alg.attrib['modelfile'] + '\",' + vec_name + ',' + timestep.text

    if 'tau_refractive' in alg.keys():
        s += ', '  + alg.attrib['tau_refractive']
    else:
        s += ', 0.0'
    if 'ratemethod' in alg.keys():
        s += ', '  + "\"" + alg.attrib['ratemethod'] + "\""
    s += ');\n'


    Register(alg.attrib['name'], cpp_name)

    return s


def parse_grid_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'GridAlgorithm':
        raise ValueError

    timestep = alg.find('TimeStep')

    cpp_name = 'alg_mesh_' + str(i)
    s += '\tTwoDLib::GridAlgorithm ' + cpp_name + '(\"'
    s += alg.attrib['modelfile'] + '\",'
    s += '\"' + alg.attrib['transformfile'] + '\",'
    s += timestep.text + ','
    s += alg.attrib['start_v'] + "," + alg.attrib['start_w']

    if 'tau_refractive' in alg.keys():
        s += ', '  + alg.attrib['tau_refractive']
    else:
        s += ', 0.0'
    if 'ratemethod' in alg.keys():
        s += ', '  + "\"" + alg.attrib['ratemethod'] + "\""
    s += ');\n'


    Register(alg.attrib['name'], cpp_name)

    return s

def parse_grid_jump_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'GridJumpAlgorithm':
        raise ValueError

    timestep = alg.find('TimeStep')

    cpp_name = 'alg_mesh_' + str(i)
    s += '\tTwoDLib::GridJumpAlgorithm ' + cpp_name + '(\"'
    s += alg.attrib['modelfile'] + '\",'
    s += '\"' + alg.attrib['transformfile'] + '\",'
    s += timestep.text + ','
    s += alg.attrib['start_v'] + "," + alg.attrib['start_w']

    if 'tau_refractive' in alg.keys():
        s += ', '  + alg.attrib['tau_refractive']
    else:
        s += ', 0.0'
    if 'ratemethod' in alg.keys():
        s += ', '  + "\"" + alg.attrib['ratemethod'] + "\""
    s += ');\n'


    Register(alg.attrib['name'], cpp_name)

    return s

def parse_grid_soma_dendrite_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'GridSomaDendriteAlgorithm':
        raise ValueError

    timestep = alg.find('TimeStep')

    cpp_name = 'alg_mesh_' + str(i)
    s += '\tTwoDLib::GridSomaDendriteAlgorithm ' + cpp_name + '(\"'
    s += alg.attrib['modelfile'] + '\",'
    s += '\"' + alg.attrib['transformfile'] + '\",'
    s += timestep.text + ','
    s += alg.attrib['start_v'] + "," + alg.attrib['start_w']

    if 'tau_refractive' in alg.keys():
        s += ', '  + alg.attrib['tau_refractive']
    else:
        s += ', 0.0'
    if 'ratemethod' in alg.keys():
        s += ', '  + "\"" + alg.attrib['ratemethod'] + "\""
    s += ');\n'


    Register(alg.attrib['name'], cpp_name)

    return s

def parse_ou_algorithm(alg, i,  weighttype):
    s = ''

    algorithmname=alg.find('OUAlgorithm')
    d=alg.attrib

    np=alg.find('NeuronParameter')
    str_np = parse_neuron_parameter(np,i)
    s += str_np

    cpp_name = 'alg_ou_' + str(i)
    s += '\tGeomLib::OUAlgorithm ' + cpp_name +'(par_neur_' + str(i) +');\n'
    Register(d['name'], cpp_name)

    return s


def parse_wilsoncowan_parameter(alg, i, weighttype):
    wcpar= alg.find('WilsonCowanParameter')
    s = ''

    t_mem = wcpar.find('t_membrane')
    t_name = 't_mem_' + str(i)
    s += '\tconst MPILib::Time ' + t_name
    s += ' = ' + t_mem.text +';\n'

    f_noise = wcpar.find('f_noise')
    noise_name ='f_noise_' + str(i)
    s += '\tconst double ' + noise_name
    s += ' = ' + f_noise.text +';\n'

    f_max = wcpar.find('f_max')
    f_name = 'f_max_' + str(i)
    s += '\tMPILib::Rate ' + f_name
    s += ' = ' + f_max.text + ';\n'

    I_ext = wcpar.find('I_ext')
    I_name = 'I_ext_' + str(i)
    s += '\tMPILib::Rate ' + I_name
    s += ' = ' + I_ext.text + ';\n'

    s += '\tMPILib::WilsonCowanParameter  par_wil_' + str(i) +'('
    s += t_name +','
    s += f_name + ','
    s += noise_name  + ','
    s += I_name  +');\n'

    return s

def wrapup_wilsoncowan_algorithm(alg,i,weighttype):
    d=alg.attrib

    if not 'name' in d.keys():
        raise NameError('Name tag expected in WilsonCowanAlgorithm')

    s = '\tMPILib::WilsonCowanAlgorithm '
    s += 'alg_wc_' + str(i) + '('
    s += 'par_wil_' + str(i)
    s += ');\n'
    cpp_name = 'alg_wc_' + str(i)
    Register(d['name'], cpp_name)
    return s

def parse_wilsoncowan_algorithm(alg, i, weighttype):
    s = ''

    s += parse_wilsoncowan_parameter(alg, i, weighttype)
    s += wrapup_wilsoncowan_algorithm(alg, i, weighttype)
    return s

def parse_persistant_algorithm(alg, i, weighttype):
    s = ''

    dp = alg.find('PersistantAlgorithm')

    if not 'Name' in dp.keys():
        raise NameError('Name tag expected')
    cpp_name = 'pers_alg_'  + str(i)
    Register(dp.attrib['Name'],cpp_name)
    s += '\tMPILib::PersistantAlgorithm ' + cpp_name + ';\n'
    return s

def parse_delay_algorithm(alg,i,weighttype):
    s = ''
    if alg.attrib['type'] != 'DelayAlgorithm':
        raise ValueError

    cpp_name = 'delay_alg_' + str(i)
    Register(alg.attrib['name'],cpp_name)

    s += '\tMPILib::algorithm::DelayAlgorithm<' + weighttype.text + '> ' + cpp_name +  '('

    dt = alg.find('delay')
    s += dt.text

    s += ');\n'

    return s

def parse_ratefunctor_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'RateFunctor':
        raise ValueError

    cpp_name = 'rate_functor_' + str(i)
    Register(alg.attrib['name'],cpp_name)


    s += '\tMPILib::Rate RateFunction_' + str(i) + '(MPILib::Time);\n'

    s += '\tMPILib::RateFunctor<' + weighttype.text + '> ' + cpp_name + '('
    s += 'RateFunction_' + str(i) + ');\n'

    cd = alg.find('code')
    rb = alg.find('expression')

    if (cd != None and rb != None):
        raise ValueError('You cannot use expression and code tags in the same RateFunctor.')
    if cd == None  and rb == 0:
        raise ValueError('You must use either a code or an expression tag in a rate functor.')
    if rb != None:
        body=rb.text
        t = 'MPILib::Rate RateFunction_' + str(i) + '(MPILib::Time t){\n'
        t += '\treturn ' + body + ';\n}\n'

    if cd != None:
        t = 'MPILib::Rate RateFunction_' + str(i) + '(MPILib::Time t){\n'
        t += cd.text
        t += '}\n\n'

    RATEFUNCTIONS.append(t)

    return s

def parse_miindmodel_ratefunctor_algorithm(alg, i, weighttype):
    s = ''
    if alg.attrib['type'] != 'RateFunctor':
        raise ValueError

    cpp_name = 'rate_functor_' + str(i)
    Register(alg.attrib['name'],cpp_name)

    s += '\tMPILib::RateFunctor<' + weighttype.text + '> ' + cpp_name + '('
    s += 'MiindModel::RateFunction_' + str(i) + ');\n'

    cd = alg.find('code')
    rb = alg.find('expression')

    if (cd != None and rb != None):
        raise ValueError('You cannot use expression and code tags in the same RateFunctor.')
    if cd == None  and rb == 0:
        raise ValueError('You must use either a code or an expression tag in a rate functor.')
    if rb != None:
        body=rb.text
        t = 'static MPILib::Rate RateFunction_' + str(i) + '(MPILib::Time t){\n'
        t += '\treturn ' + body + ';\n}\n'

    if cd != None:
        t = 'static MPILib::Rate RateFunction_' + str(i) + '(MPILib::Time t){\n'
        t += cd.text
        t += '}\n\n'

    RATEFUNCTIONS.append(t)

    return s

def parse_algorithm(alg,i,weighttype,for_lib=False):
    algname=alg.get('type')

    if algname=='WilsonCowanAlgorithm':
        if weighttype.text == 'double':
            return parse_wilsoncowan_algorithm(alg,i,weighttype)
        else:
            raise NameError('Wrong type for WilsonCowanAlgorithm')
    if algname == 'RateFunctor' and not for_lib:
        return parse_ratefunctor_algorithm(alg,i,weighttype)
    if algname == 'RateFunctor' and for_lib:
        return parse_miindmodel_ratefunctor_algorithm(alg,i,weighttype)
    if algname == 'DelayAlgorithm':
        return parse_delay_algorithm(alg,i,weighttype)
    if algname == 'PersistantAlgorithm':
        return parse_persistant_algorithm(alg,i,weighttype)
    if algname =='RateAlgorithm':
        return  parse_rate_algorithm(alg,i,weighttype)
    if algname =='OUAlgorithm':
        return  parse_ou_algorithm(alg,i,weighttype)
    if algname =='GeomAlgorithm':
        if weighttype.text == 'DelayedConnection':
            return parse_geom_algorithm(alg,i,weighttype)
        else:
            raise NameError('Wrong conection type for GeomAlgorithm')
    if algname =='MeshAlgorithm':
        if weighttype.text == 'DelayedConnection':
            return parse_mesh_algorithm(alg,i,weighttype)
        else:
            raise NameError('Wrong conection type for MeshAlgorithm')
    if algname =='MeshAlgorithmCustom':
        if weighttype.text == 'CustomConnectionParameters':
            return parse_mesh_algorithm_custom(alg,i,weighttype)
        else:
            raise NameError('Wrong conection type for MeshAlgorithmCustom')
    if algname =='GridAlgorithm':
        if weighttype.text == 'CustomConnectionParameters':
            return parse_grid_algorithm(alg,i,weighttype)
        else:
            raise NameError('Connection type for GridAlgorithm must be CustomConnectionParameters')
    if algname =='GridJumpAlgorithm':
        if weighttype.text == 'CustomConnectionParameters':
            return  parse_grid_jump_algorithm(alg,i,weighttype)
        else:
            raise NameError('Connection type for GridJumpAlgorithm must be CustomConnectionParameters')
    if algname =='GridSomaDendriteAlgorithm':
        if weighttype.text == 'CustomConnectionParameters':
            return  parse_grid_soma_dendrite_algorithm(alg,i,weighttype)
        else:
            raise NameError('Connection type for GridSomaDendriteAlgorithm must be CustomConnectionParameters')
    else:
        raise NameError('Wrong algorithm name')
    return ''

def parse_algorithms(alg_list,weighttype,outfile,for_lib=False):
    '''alg_list is a list of Algorithm elements as found by the parser. Each element
    of the list represents an individual algorithm.'''
    for i, alg in enumerate(alg_list):
        s = parse_algorithm(alg,i, weighttype,for_lib)
        outfile.write(s)
    return

def reset_algorithms():
    ALGORITHM_NAMES.clear()
    del RATEFUNCTIONS[:]

if __name__ == "__main__":
    alg_list, weighttype = setup_test()
    outfile=open('alg.cpp','w')
    parse_algorithms(alg_list,weighttype,outfile)
