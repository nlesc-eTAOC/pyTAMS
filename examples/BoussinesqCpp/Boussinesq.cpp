#include "Boussinesq.h"
#include <iomanip>

int main() {
    int M = 40, N = 80;
    double dt = 0.005;

    std::cout << std::setprecision(6);

    Boussinesq2D Boussinesq(M, N, dt);
    std::cout << "Boussinesq model initialized with M = " << M << ", N = " << N << std::endl;

    Boussinesq.advance_trajectory(20.0);

    return 0;
}
