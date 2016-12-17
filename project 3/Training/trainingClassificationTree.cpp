#include <iostream>
#include <armadillo>
#include <cmath>
#include <string>
#include "optTree.h"

using namespace std;
using namespace arma;

int main()
{
  string imagePatchFilename = "imagePatches.mat";
  string patchLabelFilename = "patchLabel.mat";
  
  mat patches;
  patches.load(imagePatchFilename, hdf5_binary);

  vec label;
  label.load(patchLabelFilename, hdf5_binary);

  int classNum =  200;
  int patchWidth = sqrt(patches.n_rows);

  int depth = 5;

  Node * tree = new Node [(3^(depth+1) - 1)/2];

  cout << label << endl;
  
  return 0;
}