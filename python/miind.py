#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import directories3 as directories
from collections import Counter

# These algorithms can feature in a MeshAlgorithmGroup simulation, and no others
MESH_ALGORITHM_GROUP_LIST = ['MeshAlgorithmGroup', 'DelayAlgorithm', 'RateFunctor' ]

def parse_rate_functors(algorithms):
     s=''
     for algorithm in algorithms:
          if algorithm.attrib['type'] == 'RateFunctor':
             expression = algorithm.find('expression')
             name=re.sub(r'\s+', '_', algorithm.attrib['name']) # a user may use white space, we need to replace it
             s += 'MPILib::Rate ' + name + '( MPILib::Time t ){\n'
             s += '\treturn ' + expression.text + ';\n'
             s += '}\n\n'
     return s

def generate_fill_in_rate_function(cuda):
     '''Create the C++ function to read the firing rates, both external and from the MeshAlgorithmGroup into the input firing rate array.'''

     if cuda == True:
          group_argument = '\tconst CudaTwoDLib::CudaOde2DSystemAdapter& group,\n'
          template_argument = 'fptype'

     else:
          group_argument = '\tconst TwoDLib::Ode2DSystemGroup& group,\n'
          template_argument = 'MPILib::Rate'


     s = ''    
     s += 'typedef MPILib::Rate (*function_pointer)(MPILib::Time);\n'
     s += 'typedef std::pair<MPILib::Index, function_pointer> function_association;\n'
     s += 'typedef std::vector<function_association> function_list;\n\n' \
     + 'void FillInRates\n' \
     + '(\n' \
     + group_argument \
     + '\tconst function_list& functor_list,\n' \
     + '\tconst std::vector<MPILib::Index>& mag_id_to_node_id,\n' \
     + '\tstd::vector<' + template_argument +'>& vec_activity_rates,\n' \
     + '\tMPILib::Time t\n' \
     + ')\n' \
     + '{\n'\
     + '\tfor(auto& rate: vec_activity_rates)\n\t\trate=0.0;\n' \
     + '\tfor( const auto& element: functor_list)\n' \
     + '\t\tvec_activity_rates[element.first] = element.second(t);\n' \
     + '\tconst std::vector<' + template_argument + '>& magrates = group.F();\n' \
     + '\tMPILib::Index i_counter = 0;\n' \
     + '\tfor (auto& rate: magrates)\n' \
     + '\t\tvec_activity_rates[mag_id_to_node_id[i_counter++]] += rate;\n' \
     + '}\n\n'

     return s;

def generate_apply_network_function(nodes,algorithms,connections,cuda):
     '''Generate the connectivity application function in the C++'''
     map = construct_CSR_map(nodes,algorithms,connections)
     magmap=node_name_to_mag_id(nodes,algorithms)
     nodemap=node_name_to_node_id(nodes)

     if cuda == True:
          template_argument = 'fptype'
     else:
          template_argument = 'MPILib::Rate'

     s = '' 
     s += 'void ApplyNetwork\n'
     s += '(\n'
     s += '\tconst std::vector<' + template_argument + '>& vec_node_rates,\n'
     s += '\tstd::vector<' + template_argument + '>& vec_mag_rates\n'
     s += '){\n'

     for i,el in enumerate(map):
          items=connections[i].text.split()
          s += '\tvec_mag_rates[' + str(i) + ']=' + items[0] + '*vec_node_rates[' + str(nodemap[el[3]]) + '];\n'
     s += '}\n\n'
     return s


def generate_variable_declarations(variables):
     s = ''
     for variable in variables:
          s += '\tconst float ' + variable.attrib['Name'] + ' = ' + variable.text + ';\n'
     s += '\n'
     return s

def generate_functor_table(nodes, algorithms):
     '''Generate a table linking the nodes corresponding to external inputs'''
     s = ''

     pairlist = []
     for i, node in enumerate(nodes):
          namealg = node.attrib['algorithm']
          for alg in algorithms:
               if alg.attrib['name'] == namealg and alg.attrib['type'] == 'RateFunctor':
                    pairlist.append((i, re.sub(r'\s+', '_', alg.attrib['name'])))

     s += '\tfunction_list functor_list;\n'
     for el in pairlist: 
          s += '\tfunctor_list.push_back(function_association(' + str(el[0]) + ',' + el[1] + '));\n'
     
     s += '\n'

     return s

def generate_log_function(cuda):
     '''Generate the code for writing simulation data into the C++ file.'''

     if cuda == True:
          template_argument = 'fptype'
     else:
          template_argument = 'MPILib::Rate'

     s = ''

     s += 'void LogData(std::ostream& s, MPILib::Time t, const std::vector<'+template_argument+'>& vec_rates)\n'
     s += '{\n'
     s += '\tMPILib::Index i_pop = 0;\n'
     s += '\tfor(auto rate: vec_rates)\n'
     s += '\t\ts << i_pop++ << "\\t" << t << "\\t" << rate << \"\\n\";\n'
     s += '}\n'
     s += '\n'
     return s

def generate_mag_table(nodes,algorithms):
     '''Generates a mapping from MeshAlgorithmGroup order to NodeId ( a MagId table).'''
     s = ''
     magtab=[]
     for i, node in enumerate(nodes):
          algorithmname = node.attrib['algorithm']
          for alg in algorithms:
               if alg.attrib['name'] == algorithmname and alg.attrib['type'] == "MeshAlgorithmGroup":
                    magtab.append(i)

     s += '\tstd::vector<MPILib::Index>  mag_id_to_node_id{\n'

     for nodeid in magtab[:-1]:
          s += '\t\t' + str(nodeid) + ',\\\n'

     nodeid=magtab[-1]
     s += '\t\t' + str(nodeid) + '\n'
     s += '\t};\n'

     s += '\n'

     return s



def generate_preamble(fn, variables, nodes, algorithms, connections, cuda):
    '''Generates the function declarations, required for RateFunctors etc in the C++ file. fn is the file name where the C++
    is to be written. variable, nodes and algorithms are XML elements.'''

    # the rate functor functions need to be declared before the main program
    function_declarations = parse_rate_functors(algorithms)
    log_function = generate_log_function(cuda)
    variable_declarations = generate_variable_declarations(variables)
    functor_table = generate_functor_table(nodes,algorithms)
    mag_table  = generate_mag_table(nodes,algorithms)

    fill_in_function = generate_fill_in_rate_function(cuda)
    apply_network_function = generate_apply_network_function(nodes, algorithms, connections, cuda)

    if cuda == True:
         template_argument = 'fptype'
    else:
         template_argument = 'MPILib::Rate'

    with open(fn,'w') as f:
        f.write('//Machine-generated by miind.py. Edit at your own risk.\n\n')
        f.write('#include <boost/timer/timer.hpp>\n')
        f.write('#include <GeomLib.hpp>\n')
        f.write('#include <TwoDLib.hpp>\n')
        if cuda == True: f.write('#include <CudaTwoDLib.hpp>\n')
        f.write('#include <MPILib/include/RateAlgorithmCode.hpp>\n')
        f.write('#include <MPILib/include/SimulationRunParameter.hpp>\n')
        f.write('#include <MPILib/include/DelayAlgorithmCode.hpp>\n')
        f.write('#include <MPILib/include/RateFunctorCode.hpp>\n\n')
        if cuda == True: f.write('typedef CudaTwoDLib::fptype fptype;\n')
        f.write(fill_in_function)
        f.write(apply_network_function)
        f.write(function_declarations)
        f.write(log_function)
        f.write('\nint main(int argc, char *argv[]){\n')
        f.write(variable_declarations)
        f.write('\tconst MPILib::Number n_populations = ' + str(len(nodes)) + ';\n\n')
        f.write(functor_table)
        f.write(mag_table)
        f.write('\tstd::vector<' + template_argument + '> vec_activity_rates(n_populations,0.0);\n')


        f.write('\n')

def generate_closing(fn):
    '''Generates the closing statements in the C++ file.'''
    with open(fn,'a') as f:
        f.write('\n')
        f.write('\tstd::cout << \"Overall time spend\";\n')
        f.write('\ttimer.report();\n')
        f.write('\treturn 0;\n')
        f.write('}\n')

def process_tree(root):

    variables=root.findall(".//Variable")
    nodes=root.findall('.//Node')
    algorithms=root.findall('.//Algorithm')
    connections=root.findall('.//Connection')
    parameters=root.findall('.//SimulationRunParameter')
    io=root.findall('.//SimulationIO')
    return variables, nodes, algorithms, connections, parameters, io

def parse(fn):
    '''Takes a filename. Puts the file with filename fn through the XML parser. Returns nothing.'''
    try:
        tree = ET.parse(fn)
        root = tree.getroot()

    except FileNotFoundError:
        print('No file ' + fn)
    return root

def generate_simulation_parameter(fn, parameters):
     '''Write the simulation parameter into the simulation file specified by file name fn.'''
     t_start_els = parameters[0].findall('t_begin')
     t_start = t_start_els[0].text

     t_end_els = parameters[0].findall('t_end')
     t_end = t_end_els[0].text
     
     t_report_els = parameters[0].findall('t_report')
     t_report = t_report_els[0].text

     t_network_els = parameters[0].findall('t_step')
     t_network = t_network_els[0].text

     with open(fn,'a') as f:
          f.write('\n')
          f.write('\tMPILib::Time t_begin = ' + t_start + ';\n')
          f.write('\tMPILib::Time t_end = ' + t_end + ';\n')
          f.write('\tMPILib::Time t_report = ' + t_report + ';\n')
          f.write('\tMPILib::Time t_step = ' + t_network + ';\n\n')
          f.write('\tMPILib::Number n_iter = static_cast<MPILib::Number>(ceil((t_end - t_begin)/t_step));\n')
          f.write('\tMPILib::Number n_report = static_cast<MPILib::Number>(ceil((t_report - t_begin)/t_step));\n')


def generate_model_files(nodes,algorithms):
     modelfiles=[]
     matrixfilelist=[]
     for node in nodes:
          algname = node.attrib['algorithm']
          for alg in algorithms:
               if alg.attrib['name'] == algname: # here we assume the name is unique
                    algorithm = alg

          if algorithm.attrib['type'] == 'MeshAlgorithmGroup':
               modelfiles.append(algorithm.attrib['modelfile'])
               ts = algorithm.findall('TimeStep')
               timestep = float(ts[0].text)

     return modelfiles, timestep

def generate_mesh_algorithm_group(fn,nodes,algorithms,cuda):
     '''Colate al MeshAlgorithmGroup instances and generate the C++ code to instantiate the group'''

     modelfiles, timestep = generate_model_files(nodes,algorithms)

     with open(fn,'a') as f:
          f.write('\tpugi::xml_document doc;\n')
          f.write('\tstd::vector<TwoDLib::Mesh> vec_vec_mesh;\n')
          f.write('\tstd::vector< std::vector<TwoDLib::Redistribution> > vec_vec_rev;\n')
          f.write('\tstd::vector< std::vector<TwoDLib::Redistribution> > vec_vec_res;\n');
          for i,model in enumerate(modelfiles):
               # according to pugixml doc, load_file destroys the old tree, soo this should be save
               f.write('\tpugi::xml_parse_result result' + str(i) + ' = doc.load_file(\"' + model +'\");\n')
               f.write('\tpugi::xml_node  root' + str(i) + ' = doc.first_child();\n\n')
               f.write('\tTwoDLib::Mesh mesh' + str(i) +' = TwoDLib::RetrieveMeshFromXML(root' + str(i) + ');\n')
               f.write('\tstd::vector<TwoDLib::Redistribution> vec_rev' + str(i) + ' = TwoDLib::RetrieveMappingFromXML("Reversal",root' + str(i) + ');\n')
               f.write('\tstd::vector<TwoDLib::Redistribution> vec_res' + str(i) + ' = TwoDLib::RetrieveMappingFromXML("Reset",root' + str(i) + ');\n\n')
               f.write('\tvec_vec_mesh.push_back(mesh'+str(i)+');\n')
               f.write('\tvec_vec_rev.push_back(vec_rev'+str(i)+');\n')
               f.write('\tvec_vec_res.push_back(vec_res'+str(i)+');\n')
          
     
          f.write('\tTwoDLib::MasterParameter par(' + 'static_cast<MPILib::Number>(ceil(mesh0.TimeStep()/' + str(timestep) + ')));\n\n')
          f.write('\tconst MPILib::Time h = 1./par._N_steps*mesh0.TimeStep();\n')
          


def node_name_to_node_id(nodes):
     '''Create a map from name to NodeId from node elements. Return this map.'''
     d ={}
     for i,node in enumerate(nodes):
          d[node.attrib['name']] = i
     return d

def extract_efficacy(fn):
     '''Extract efficacy from a matrix file. Takes a filename, returns efficacy as a single float. In the
     file efficacies are represented by two numbers. We will assume for now that one of them in zero. We will
     return the non-zero number as efficacy.'''

     with open(fn) as f:
          line=f.readline()
          nrs = [ float(x) for x in line.split()]
          if nrs[0] == 0.:
               return nrs[1]
          else:
               if nrs[1] != 0:
                    raise ValueError('Expected at least one non-zero value')
               return nrs[0]



def node_id_to_mag_id(nodes, algorithms):
     '''MagId is determined by the node order. Every time a MeshAlgorithmGroup is encoutered when traversing nodes in their  order in the XML file, the MagId
     is increased.'''
     map = []
     nodecounter = 0
     magcounter = 0
     for node in nodes:
          algname = node.attrib['algorithm']
          for algorithm in algorithms:
               if algorithm.attrib['name'] == algname:
                    if algorithm.attrib['type'] == 'MeshAlgorithmGroup':
                         map.append([[nodecounter],[magcounter]])
                         magcounter += 1
          nodecounter += 1
     return map

def node_name_to_mag_id(nodes,algorithms):
     '''Map a node name to the correct MagId, if it exists.'''
     map = {}
     magcounter = 0
     for node in nodes:
          algname = node.attrib['algorithm']
          for algorithm in algorithms:
               if algorithm.attrib['name'] == algname:
                    if algorithm.attrib['type'] == 'MeshAlgorithmGroup':
                         map[node.attrib['name']]  = magcounter
                         magcounter += 1
     return map


def construct_CSR_map(nodes,algorithms,connections):
     '''Creates a list that corresponds one-to-one with the connection structure. Returns a tuple: [0] node name of receiving node,[1] matrix file name for this connection  '''
     csrlist=[]
     combi = []
     for connection in connections:
          for node in nodes:
               if connection.attrib['Out'] == node.attrib['name']:
                    # we have the right node, now see if it's a MeshAlgorithmGroup
                    nodealgorithm=node.attrib['algorithm']
                    for algorithm in algorithms:
                         if nodealgorithm == algorithm.attrib['name']:
                              if algorithm.attrib['type'] == 'MeshAlgorithmGroup':
               

                                   mfs=algorithm.findall('MatrixFile')
                                   mfn= [ mf.text for mf in mfs] 
                                   efficacy=float(connection.text.split()[1])
                                   effs= [extract_efficacy(fn) for fn in mfn]
                    
                                   candidates=[]
                                   for i, eff in enumerate(effs):
                                        if np.isclose(eff,efficacy):
                                             candidates.append(i)
                                   if len(candidates) == 0: raise ValueError('No efficacy found that corresponds to the connection efficacy ' + str(efficacy))
                                   if len(candidates) > 1: raise ValueError('Same efficacy found twice')

                                   count = Counter(combi)
                                   combi.append((connection.attrib['Out'],connection.attrib['In']))
                                   nr_connection = count[(connection.attrib['Out'],connection.attrib['In'])]
                                   csrlist.append([node.attrib['name'],mfn[candidates[0]], effs[candidates[0]],connection.attrib['In'],nr_connection])
                                   
     return csrlist

                                                                      
def generate_connectivity(fn, nodes, algorithms, connections,cuda):
     '''Write out the CSR matrix lists and vectors into the C++ file.'''
     map = construct_CSR_map(nodes,algorithms,connections)
     nodemap=node_name_to_node_id(nodes)
     magmap=node_name_to_mag_id(nodes,algorithms)

     with open(fn,'a') as f:
          for el in map:
               f.write('\tTwoDLib::TransitionMatrix mat_' + str(nodemap[el[0]]) + '_' + str(nodemap[el[3]]) + '_' + str(el[4]) + '(\"' + el[1] + '\");\n')  
          f.write('\tconst std::vector<TwoDLib::CSRMatrix> vecmat {\\\n')

          if cuda == True:
               template_argument = 'fptype'
               group = ', group_ode,'
          else:
               group = ', group,'
               template_argument = 'MPILib::Rate'

          for el in map[:-1]:
               f.write('\t\tTwoDLib::CSRMatrix(mat_' + str(nodemap[el[0]]) + '_' + str(nodemap[el[3]])+ '_' + str(el[4]) + group +  str(magmap[el[0]]) + '), \\\n')
          el = map[-1]
          f.write('\t\tTwoDLib::CSRMatrix(mat_' + str(nodemap[el[0]]) + '_' + str(nodemap[el[3]])+ '_' + str(el[4]) + group +  str(magmap[el[0]]) + ') \\\n')
          f.write('\t};\n')
          f.write('\tstd::vector<' + template_argument + '> vec_magin_rates(vecmat.size(),0.);\n\n')
          f.write('\n')
          if cuda:
               f.write('\tCudaTwoDLib::CSRAdapter csr_adapter(group,vecmat,h);')
          else:
               f.write('\tTwoDLib::CSRAdapter csr_adapter(group,vecmat,h);')

def generate_simulation_loop(fn,cuda):
     '''Write the simulation loop into the C++ file.'''
     with open(fn,'a') as f:
          f.write('\tMPILib::Time time = 0;\n')
          f.write('\tboost::timer::auto_cpu_timer timer;\n')
          f.write('\tfor(MPILib::Index i_loop = 0; i_loop < n_iter; i_loop++){\n')
          f.write('\t\ttime = t_step*i_loop;\n')
          f.write('\t\tApplyNetwork(vec_activity_rates,vec_magin_rates);\n')
          f.write('\n\t\tgroup.Evolve();\n')
          f.write('\t\tgroup.RemapReversal();\n')
          f.write('\t\tfor (MPILib::Index i_part = 0; i_part < par._N_steps; i_part++ ){\n')
          f.write('\t\t\tcsr_adapter.ClearDerivative();\n')
          f.write('\t\t\tcsr_adapter.CalculateDerivative(vec_magin_rates);\n')
          f.write('\t\t\tcsr_adapter.AddDerivative();\n')
          f.write('\t\t}\n')
          f.write('\t\tgroup.RedistributeProbability();\n')
          if cuda:
               f.write('\t\tgroup.MapFinish();\n')
          f.write('\t\tFillInRates(group, functor_list, mag_id_to_node_id, vec_activity_rates, time);\n')
          f.write('\t\tif (i_loop%n_report == 0) LogData(log,time,vec_activity_rates);\n')
          f.write('\t}\n')

def generate_simulation_io(fn,io):
     '''Write I/O paremeters into C++ file.'''
     logname=io[0].findall('SimulationName') # just one io parameter
     with open(fn,'a') as f:
          f.write('\tstd::ofstream log("' + logname[0].text + '");\n')

def generate_initialization(fn,nodes,algorithms,cuda):
     '''Create C++ obejcts to be used inside the loop and initialize the Ode2DSystemGroup.'''
     with open(fn,'a') as f:

          if (cuda):
               group = 'group_ode'
               # in cuda, the group actions are performed by an adapter. To maintain identical code, the adpater is called group
          else:
               group = 'group'


          f.write('\tTwoDLib::Ode2DSystemGroup ' + group + '(vec_vec_mesh,vec_vec_rev, vec_vec_res);\n\n')
          f.write('\tMPILib::Index i_mesh = 0;\n')
          f.write('\tfor( auto id : mag_id_to_node_id)\n')
          f.write('\t\t' + group + '.Initialize(i_mesh++,0,0);\n')
          if cuda: f.write('\tCudaTwoDLib::CudaOde2DSystemAdapter group(group_ode);\n\n')
          f.write('\n\n')

     
     

def create_cpp_file(xmlfile, dirpath, progname, modname, cuda):
    '''Write the C++ file specified by xmlfile into dirpath as progname.'''
    root=parse(xmlfile)
    variables, nodes, algorithms, connections, parameter, io=process_tree(root)
    if sanity_check(algorithms) == False: raise NameError('An algorithm incompatible with MeshAlgorithmGroup was used')
    if cuda == True:
         fn=os.path.join(dirpath, progname)+'.cu'
    else:
         fn=os.path.join(dirpath, progname)+'.cpp'

    generate_preamble(fn, variables, nodes, algorithms,connections,cuda)
    generate_mesh_algorithm_group(fn,nodes,algorithms,cuda)
    generate_initialization(fn,nodes,algorithms,cuda)
    generate_connectivity(fn,nodes,algorithms,connections,cuda)
    generate_simulation_parameter(fn,parameter)
    generate_simulation_io(fn,io)
    generate_simulation_loop(fn,cuda)
    generate_closing(fn)

def sanity_check(algorithms):
    '''Check if only the allowd algorithms feature in this simulation. Returns True if so, False otherwise.'''

    for algorithm in algorithms:
        if algorithm.attrib['type'] not in MESH_ALGORITHM_GROUP_LIST:
            return False
        else:
            return True

def mesh_algorithm_group(root):
    '''True if there are MeshAlgorithmGroup algorithms in the XML file, false otherwise.'''
    algorithms=root.findall('.//Algorithm')

    for algorithm in algorithms:
        if algorithm.attrib['type'] == "MeshAlgorithmGroup":
            return True

    return False

def produce_mesh_algorithm_version(dirname, filename, modname, root, cuda):
    '''Entry point for the vector version of a MIIND C++ file. Filename is file name of the XML file, dirname is the user-specified directory hierarchy
    where the C++ file will be generated and the simulation will be stored. The simulation file will be placed in directory <dirname>/<xml_file_name>.'''

    if not directories.PATH_VARS_DEFINED:
        directories.initialize_global_variables()

    for xmlfile in filename:
        progname = directories.check_and_strip_name(xmlfile)
        dirpath = directories.create_dir(os.path.join(dirname, progname))
        directories.insert_cmake_template(progname,dirpath,cuda)
        create_cpp_file(xmlfile, dirpath, progname, modname, cuda)
        directories.move_model_files(xmlfile,dirpath)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate C++ from XML descriptions.')
    parser.add_argument('--d', help = 'Provide a packaging directory.',nargs = '?')
    parser.add_argument('-c', '--cuda', action="store_true", dest="cuda", help="if specified, cuda will be generated")
    parser.add_argument('-m','--m', help = 'A list of model and matrix files that will be copied to every executable directory.''', nargs='+')
    parser.add_argument('xml file', metavar='XML File', nargs = '*', help = 'Will create an entry in the build tree for each XML file, provided the XML file is valid.')
    args = parser.parse_args()


    filename = vars(args)['xml file']
    dirname  = vars(args)['d']
    modname  = vars(args)['m']

    
    fn = filename[0]
    root=parse(fn)
    if mesh_algorithm_group(root) == True:
        # Run the MeshAlgorithm version
        produce_mesh_algorithm_version(dirname, filename, modname, root, vars(args)['cuda'])
    else:
        # Simply run the old script
        if dirname == None:
            fn = filename[0]
            directories.add_executable(fn,modname)
        else:
            directories.add_executable(dirname, filename, modname)

