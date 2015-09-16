#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

std::random_device generator;
std::gamma_distribution<double> gamma_dist;


void iniRandom(double alpha, double beta){
  gamma_dist=std::gamma_distribution<double>(alpha, beta);
}


double gamma_rand(void){
  return gamma_dist(generator);
}
