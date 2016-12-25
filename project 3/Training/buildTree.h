#ifndef BUILD_TREE
#define BUILD_TREE
#include "optTree.h"

// build tree with recursive
// parameters: mat& patches - (subset) training image data
//			   vec& label - (subste) training image label
//			   int classNum - number of classes
//		       int patchWidth - width of patch
//			   int depth - depth of the tree
//			   int currentDepth - current depth of tree
//			   TreeNode * finalTree - node of tree; store all information of the tree
int treebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree);

// travel whole tree and write the data in every node to file
// parameters: fstream& file - reference of fstream object
//			   int classNum - number of classes
//		       int patchWidth - width of patch
//			   int depth - depth of the tree
//			   int currentDepth - current depth of tree
//			   TreeNode * finalTree - node of tree; store all information of the tree 
int travelTree(fstream& file, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree);

// output the propability of each class based on given image
// parameters: vec& data - given testing image
//			   int classNum - number of classes
//		       int patchWidth - width of patch
//			   TreeNode * finalTree - node of tree; store all information of the tree 
vec * testTree(vec& data, int classNum, int patchWidth, TreeNode * finalTree);

#endif