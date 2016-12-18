#ifndef BUILD_TREE
#define BUILD_TREE
#include "optTree.h"

void fullTreebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, Node ** finalTree);
int treebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree);

#endif