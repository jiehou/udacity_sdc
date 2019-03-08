#ifndef UTILS_H_
#define UTILS_H_
#include <iostream>
using std::cout;
using std::endl;

bool ParseCommands(int argc, char* argv[], double& k_p, double& k_i, double& k_d)
{
    if(argc != 4)
    {
        cout << "#[I] please initialize PID controller in the following way: " << endl;
        cout << "#[I] " << argv[0] << "Kp(0.1) Ki(0.005) Kd(1.0)" << endl;
        return false;
    }
    k_p = atof(argv[1]);
    k_i = atof(argv[2]);
    k_d = atof(argv[3]);
    return true;
}
#endif