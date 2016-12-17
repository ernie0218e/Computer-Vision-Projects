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
  Dataset * subsets = new Dataset[3];

  optTree(patches, label, classNum, patchWidth, &tree[0], subsets);

  for (int k = 1;k <= 3;k++)
  {
    tree[k].subset = subsets[k-1];
  }
  delete []subset;
  
  int count = (3^(depth) - 1)/2;
  for (int k = 1;k < count;k++)
  {
      Dataset * subsets = new Dataset[3];
      optTree(tree[k].subset.dataIndices, tree[k].subset.label, classNum, patchWidth, &tree[k], subsets);
      
      for (int j = 1;j <= 3;j++)
      {
        tree[3*k + j].subset = subsets[j-1];
      }
      delete []subset;
  }
  
  return 0;
}