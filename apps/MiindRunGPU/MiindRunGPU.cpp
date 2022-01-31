#include <MiindLib/SimulationParserGPU.h>

// The entirety of MIIND is based on templated code which is not resolved when MIIND is built.
// It is resolved when the cpp code is generated in python and compiled by the user.
// This means that if we want to have a standalone executable or python library,
// we need to do some seriously rough hacking.
// As though we are eating roasted Ortolan, we must cover our heads in a veil to hide the shame
// from God.
// At least we know that there are only three possible Weight Types (In the future, we should only support
// CustomConnectionParameters as it is the most flexible). However, which model we need is not known
// until we look at WeightType in the XML file.
SimulationParserGPU<MPILib::CustomConnectionParameters>* modelCcp;
SimulationParserGPU<MPILib::DelayedConnection>* modelDc;

TwoDLib::Display* display;
bool display_threaded = false;

std::map<std::string, std::string> getVariablesFromFile(std::string filename) {
    std::map<std::string, std::string> dict;

    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        std::cout << "Failed to load XML simulation file.\n";
        return dict;
    }


    for (pugi::xml_node var = doc.child("Simulation").child("Variable"); var; var = var.next_sibling("Variable")) {
        dict[std::string(var.attribute("Name").value())] = std::string(var.text().as_string());
    }

    return dict;
}

void InitialiseModel(int num_nodes, std::string filename, std::map<std::string, std::string> variables) {
    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        std::cout << "Failed to load XML simulation file.\n";
        return;
    }

    if (std::string("CustomConnectionParameters") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: CustomConnectionParameters.\n";
        modelCcp = new SimulationParserGPU<MPILib::CustomConnectionParameters>(num_nodes, filename, variables);
        modelCcp->init();
    }
    else if (std::string("DelayedConnection") == std::string(doc.child("Simulation").child_value("WeightType"))) {
        std::cout << "Loading simulation with WeightType: DelayedConnection.\n";
        modelDc = new SimulationParserGPU<MPILib::DelayedConnection>(num_nodes, filename, variables);
        modelDc->init();
    }
}

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void updateDisplay(TwoDLib::Display* disp, long *iterations, double ts) {
    disp->animate(true, ts);

    while(true)
        disp->updateDisplay(*iterations);
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        std::cout << "MiindRun expects a MIIND simulation xml file as parameter 1.\n";
        return 1;
    }

    display = TwoDLib::Display::getInstance();
        
    std::string xmlfile(argv[1]);
    int node_count = 1;

    unsigned int arg_index = 2;

    if (argc >= 3 && is_number(argv[arg_index])) {
        node_count = std::stoi(argv[arg_index]);
        arg_index = 3;
    }

    std::map<std::string, std::string> vars;
    for (; arg_index < argc; arg_index++) {
        std::string arg = std::string(argv[arg_index]);
        std::string delimiter = "=";
        std::string var_name = arg.substr(0, arg.find(delimiter));
        std::string var_val = arg.substr(0, arg.find(delimiter));
        vars[var_name] = var_val;
    }

    InitialiseModel(node_count, xmlfile, vars);
    double time = 0.0;
    long its = 0;

    if (modelCcp) {
        std::cout << "Time Step: " << modelCcp->getTimeStep() << "\n";
        std::cout << "Sim Time: " << modelCcp->getSimulationLength() << "\n";
        modelCcp->startSimulation(display);

        std::thread worker_thread;
        if (display_threaded) {
            worker_thread = std::thread(updateDisplay, display, &its, modelCcp->getTimeStep());
        }
        else {
            display->animate(true, modelCcp->getTimeStep());
        }

        while (time < modelCcp->getSimulationLength()) {
            time += modelCcp->getTimeStep();
            modelCcp->evolveSingleStep(std::vector<double>());
            if (!display_threaded)
                display->updateDisplay(its);
            its++;
        }
        if (display_threaded) {
            worker_thread.join();
        }
        modelCcp->endSimulation();
    }
    else if (modelDc) {
        std::cout << "Time Step: " << modelDc->getTimeStep() << "\n";
        std::cout << "Sim Time: " << modelDc->getSimulationLength() << "\n";
        modelDc->startSimulation(display);
        std::thread worker_thread;
        if (display_threaded) {
            worker_thread = std::thread(updateDisplay, display, &its, modelDc->getTimeStep());
        }
        else {
            display->animate(true, modelDc->getTimeStep());
        }
        while (time < modelDc->getSimulationLength()) {
            time += modelDc->getTimeStep();
            modelDc->evolveSingleStep(std::vector<double>());
            if (!display_threaded)
                display->updateDisplay(its);
            its++;
        }
        if (display_threaded) {
            worker_thread.join();
        }
        modelDc->endSimulation();
    }

    

}