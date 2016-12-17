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

  int treeSize = (pow(3, (depth+1)) - 1)/2.0;

  cout << "Tree size: " << treeSize << endl;

  Node * tree = new Node [treeSize];
  Dataset * subsets = new Dataset[3];

  optTree(patches, label, classNum, patchWidth, &tree[0], subsets);

  for (int k = 1;k <= 3;k++)
  {
    tree[k].subset = subsets[k-1];
  }
  delete []subsets;
  
  int count = (pow(3, (depth)) - 1)/2.0;

  cout << "count: " << count << endl;

  for (int k = 1;k < count;k++)
  {
      cout << "k: " << k << endl;
      Dataset * subsets = new Dataset[3];
      optTree(*(tree[k].subset.data), *(tree[k].subset.label), classNum, patchWidth, &tree[k], subsets);
      
      for (int j = 1;j <= 3;j++)
      {
        tree[3*k + j].subset = subsets[j-1];
      }
      delete []subsets;
  }

  cout << "here" << endl;

  for (int k = count;k < treeSize;k++)
  {
      Dataset * subset = &tree[k].subset;
      vec * label = subset->label;

      double I = label->n_rows;
      double max = 0;
      for (int c = 1;c <= classNum;c++)
      {
          double num = 0;
          for (int l = 0;l < label->n_rows;l++)
          {
              if ((*label)(l) == c)
              {
                  num = num + 1;
              }
          }
          double lambda = num/I;

          if(lambda > max)
            max = lambda;
      }
      cout << max << endl;
  }
  
  return 0;
}