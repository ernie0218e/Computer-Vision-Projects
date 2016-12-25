#include "buildTree.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;

int treebuilder(mat& patches, vec& label, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree)
{
	int I = patches.n_cols;
	// if current depth reach the setting depth
	// or there are too few data
	if (currentDepth >= (depth) || I < 20)
	{
		// we have reached the leaf in the tree
		finalTree->node = new Node;
		finalTree->childNodes = new TreeNode *[3];
		for (int k = 0; k < 3; k++)
		{
			finalTree->childNodes[k] = nullptr;
		}

		finalTree->node->subset.data = new mat(patches);
		finalTree->node->subset.label = new vec(label);
		// save remaining data and label for calculating probability
		// terminate
		return 1;
	}
	else
	{
		// we still can generate more nodes
		Dataset * subsets = new Dataset[3];
		for (int i = 0; i < 3; i++)
		{
			subsets[i].data = nullptr;
			subsets[i].label = nullptr;
		}

		finalTree->node = new Node;
		
		// find the best way to divide date into three parts
		// and the decision rule of this node
		optTree(patches, label, classNum, patchWidth, finalTree->node, subsets, currentDepth);

		finalTree->childNodes = new TreeNode * [3];

		// in the order of left, center, right
		for (int k = 0; k < 3; k++)
		{
			finalTree->childNodes[k] = new TreeNode;

			// grow this tree recursively
			// depth first
			int val = treebuilder(*(subsets[k].data), *(subsets[k].label), classNum, patchWidth, depth,
				currentDepth + 1, finalTree->childNodes[k]);

			// delete useless data
			if (subsets[k].data != nullptr)
				delete subsets[k].data;
			if (subsets[k].label != nullptr)
				delete subsets[k].label;
		}

		delete []subsets;
		return 0;
	}
}

int travelTree(fstream& file, int classNum, int patchWidth, int depth, int currentDepth, TreeNode * finalTree)
{
	// if we have reached the leaf
	if (finalTree->childNodes[0] == nullptr)
	{
		// mark this node as a 'leaf' node
		currentDepth = -1;
		file << currentDepth << " ";
		
		// calculate posterior probability;
		double I = finalTree->node->subset.label->n_rows;
		double max = 0;
		for (int c = 1; c <= classNum; c++)
		{
			double num = 0;
			for (int l = 0; l < I; l++)
			{
				if ((*(finalTree->node->subset.label))(l) == c)
				{
					num = num + 1;
				}
			}
			double lambda = num / I;

			// output the probability to file
			file << lambda << " ";
		}
		return 1;
	}
	else
	{
		// we still can go deeper

		Point dpt_1 = finalTree->node->pt_dm1;
		Point dpt_2 = finalTree->node->pt_dm2;

		// output current depth and the decision rule of this node
		file << currentDepth << " " << dpt_1.x << " " <<  dpt_1.y << " " << dpt_2.x << " " << dpt_2.y << endl;

		// in the order of left, center, right
		for (int k = 0; k < 3; k++)
		{
			// travel all the node inside this tree
			// go to next node
			int val = travelTree(file, classNum, patchWidth, depth, currentDepth + 1, finalTree->childNodes[k]);
		}

		return 0;
	}
}

vec * testTree(vec& data, int classNum, int patchWidth, TreeNode * finalTree)
{
	// if we have reached the leaf
	if (finalTree->childNodes[0] == nullptr)
	{
		// calculate posterior probability of each class
		double I = finalTree->node->subset.label->n_rows;
		vec * lambda = new vec(classNum);
		double maxVal = 0;
		for (int c = 1; c <= classNum; c++)
		{
			double num = 0;
			for (int l = 0; l < I; l++)
			{
				if ((*(finalTree->node->subset.label))(l) == c)
				{
					num = num + 1;
				}
			}
			double temp = num / I;

			(*lambda)(c - 1) = temp;

		}
		// return the probability
		return lambda;
	}
	else
	{
		// we still can go deeper

		// get the info of decision rule from this node.
		Point dpt_1 = finalTree->node->pt_dm1;
		Point dpt_2 = finalTree->node->pt_dm2;

		dpt_1.x = dpt_1.x + (patchWidth - 1) / 2;
		dpt_1.y = dpt_1.y + (patchWidth - 1) / 2;
		dpt_2.x = dpt_2.x + (patchWidth - 1) / 2;
		dpt_2.y = dpt_2.y + (patchWidth - 1) / 2;

		// the decision rule is simply based on calculating the difference
		// of image intensity
		int diff = data(dpt_1.x*patchWidth + dpt_1.y) - data(dpt_2.x*patchWidth + dpt_2.y);

		vec * lambda;

		// decide which child node we should go.
		if (diff < -10)
		{
			lambda = testTree(data, classNum, patchWidth, finalTree->childNodes[0]);
		}
		else if (abs(diff) <= 10)
		{
			lambda = testTree(data, classNum, patchWidth, finalTree->childNodes[1]);
		}
		else
		{
			lambda = testTree(data, classNum, patchWidth, finalTree->childNodes[2]);
		}

		return lambda;
	}
}