from parametersweep import *

def no_decay():
    f=xml_file('response.xml')
    tag_mem = xml_tag('<t_membrane>50e-3</t_membrane>')
    f.replace_xml_tag(tag_mem,50e-3)

    tag_screen = xml_tag('<OnScreen>TRUE</OnScreen>')
    f.replace_xml_tag(tag_screen,'FALSE')

    tag_con = xml_tag('<Connection In="Cortical Background" Out="LIF E">800 0.03 0</Connection>')
    f.replace_xml_tag(tag_con,10000., 0)
    f.replace_xml_tag(tag_con,0.05, 1)

    tag_report = xml_tag('<t_report>1e-05</t_report>')
    tag_state  = xml_tag('<t_state_report>1e-03</t_state_report>')
    tag_update = xml_tag('<t_update>1e-05</t_update>')

    tag_time = xml_tag('<t_end>0.3</t_end>')
    f.replace_xml_tag(tag_time,1e-3)

    f.replace_xml_tag(tag_state, 1e-5)
    f.replace_xml_tag(tag_report,1e-5)
    f.replace_xml_tag(tag_update,1e-5)

    tag_bins= xml_tag('<N_bins>500</N_bins>')
    f.replace_xml_tag(tag_bins,100000)

    tag_sim_results = xml_tag('<SimulationName>response</SimulationName>')
    f.replace_xml_tag(tag_sim_results,'no_decay')

    f.write('no_decay.xml')


def no_input():
    f=xml_file('response.xml')
    tag_con = xml_tag('<Connection In="Cortical Background" Out="LIF E">800 0.03 0</Connection>')
    f.replace_xml_tag(tag_con,0.0,0)

    tag_mu    = xml_tag('<mu>0.0</mu>')
    f.replace_xml_tag(tag_mu,0.5)

    tag_sigma = xml_tag('<sigma>0.0</sigma>')
    f.replace_xml_tag(tag_sigma,0.1)

    tag_screen = xml_tag('<OnScreen>TRUE</OnScreen>')
    f.replace_xml_tag(tag_screen,'FALSE')

    tag_sim_results = xml_tag('<SimulationName>response</SimulationName>')
    f.replace_xml_tag(tag_sim_results,'no_input')


    f.write('no_input.xml')

def single():
    f=xml_file('response.xml')

    tag_screen = xml_tag('<OnScreen>TRUE</OnScreen>')
    f.replace_xml_tag(tag_screen,'FALSE')

    tag_sim_results = xml_tag('<SimulationName>response</SimulationName>')
    f.replace_xml_tag(tag_sim_results,'single')


    f.write('response.xml')


if __name__ == "__main__":
    no_decay()
    no_input()
    single()
