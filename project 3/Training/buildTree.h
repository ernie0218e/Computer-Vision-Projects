#ifndef BUILD_TREE
#define BUILD_TREE
#include "optTree.h"

void fullTreebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, Node ** finalTree);
int treebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree);
int travelTree(fstream& file, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree);
Pair testTree(vec& data, int classNum, int patchWidth, TreeNode * finalTree);
void uniqueRandom(int **value, int pointAmount, int randomMax);

#endif