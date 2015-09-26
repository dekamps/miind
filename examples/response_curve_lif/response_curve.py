import numpy as np
from parametersweep import *

THETA     = 20e-3
TAU       = 20e-3
BASE_FILE = 'response'

def generate_mu_sigma():
    tau  = 20e-3
    mu = np.arange(15e-3,21e-3,0.5e-3)
    sigma = [1e-3, 2e-3, 5e-3]
    return tau, mu, sigma


def generate_h_rate_sequence():
    l = []
    tau, mu, sigma = generate_mu_sigma()
    for m in mu:
        for s in sigma:
            h    = s*s/m
            rate = m*m/(s*s*tau)
            name = '_' + str(s) + '_' + str(m)
            l.append([h, rate, name])
    return l

def generate_xml_file(base_file_name, element):

    f_xml = xml_file(base_file_name +'.xml')
    # adapt threshold
    tag_th  = xml_tag('<V_threshold>1.0</V_threshold>')
    f_xml.replace_xml_tag(tag_th,str(THETA))
    tag_tau = xml_tag('<t_membrane>50e-3</t_membrane>')
    f_xml.replace_xml_tag(tag_tau,str(TAU))

    tag_con=xml_tag('<Connection In="Cortical Background" Out="LIF E">800 0.03 0</Connection>')
    f_xml.replace_xml_tag(tag_con,element[1],0)
    f_xml.replace_xml_tag(tag_con,element[0],1)

    tag_fn  = xml_tag('<SimulationName>response</SimulationName>')
    f_xml.replace_xml_tag(tag_fn,base_file_name +  element[2])

    tag_log = xml_tag('<name_log>response.log</name_log>')
    f_xml.replace_xml_tag(tag_log, base_file_name + '.log')

    tag_time = xml_tag('<t_end>0.3</t_end>')
    f_xml.replace_xml_tag(tag_time,1.0)

    tag_screen = xml_tag('<OnScreen>TRUE</OnScreen>')
    f_xml.replace_xml_tag(tag_screen,'FALSE')

    f_xml.write(base_file_name+element[2]+'.xml')

def generate_xml_sequence(base_file_name):
    global FILE
    l = generate_h_rate_sequence()
    for element in l:
        generate_xml_file(base_file_name, element)

if __name__ == "__main__":
    generate_xml_sequence('response')
