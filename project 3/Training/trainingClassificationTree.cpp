// Filename: trainingClassificationTree.cpp
// Purpose: traininig classification tree
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <string>
#include "optTree.h"
#include "buildTree.h"

using namespace std;
using namespace arma;

// Input: mat& patches - input training data (D x I)
//		  vec& label - input label of training data (I x 1)
//		  int classNum - number of class
//		  int patchWidth - width of each patch (sqrt(D))
//		  int depth - depth of each tree
void testTree(mat& patches, vec& label, int classNum, int patchWidth, int depth)
{
	int treeNumber = 20;

	TreeNode * roots = new TreeNode[treeNumber];

	int I = patches.n_cols;

	// build 'treeNumber' trees
	for (int t = 0; t < treeNumber; t++)
	{
		cout << "Tree number: " << t + 1 << endl;
		treebuilder(patches, label, classNum, patchWidth, depth, 0, &roots[t]);
	}

	fstream file;
	file.open("error.txt", ios::out);

	// test training error
	double errorRate = 0;
	for (int i = 0; i < I; i++)
	{
		// get 1 training data
		vec data = patches.col(i);

		// store the probability of each class
		vec lambda = zeros<vec>(classNum);
		for (int t = 0; t < treeNumber; t++)
		{
			// sum the probability of each class from different tree
			vec * tempLambda = testTree(data, classNum, patchWidth, &roots[t]);
			lambda = lambda + (*tempLambda);

			delete tempLambda;
		}

		// normalize
		lambda = lambda / treeNumber;

		// error occurs
		if ((lambda.index_max() + 1) != label(i))
		{
			errorRate++;
		}
	}

	cout << "Error rate: " << errorRate / I << endl;
	file << "Error rate: " << errorRate / I << endl;

	// load testing data
	mat org;
	org.load("imagePatches_orig.mat", hdf5_binary);
	mat label_org;
	label_org.load("patchLabel_orig.mat", hdf5_binary);

	int num = org.n_cols;

	// calculate testing error
	// also test the effect of different threshold
	double threshold = 1;
	for (int k = 0; k < 9; k++)
	{
		// test testing error
		errorRate = 0;
		for (int i = 0; i < num; i++)
		{
			vec data = org.col(i);
			vec lambda = zeros<vec>(classNum);
			for (int t = 0; t < treeNumber; t++)
			{
				vec * tempLambda = testTree(data, classNum, patchWidth, &roots[t]);
				// sum the probability of each class from different tree
				lambda = lambda + (*tempLambda);
				delete tempLambda;

			}

			lambda = lambda / treeNumber;

			// sort probability of each class
			uvec indices = sort_index(lambda, "descend");

			// if (maximum value*threshold < second large value) --> (view that patch as background)
			//	or (the index of maximum value != real label)
			// error occurs
			if (lambda(indices(1)) > threshold*lambda(indices(0)) || (((int)indices(0)+1) != label_org(i)))
			{
				errorRate = errorRate + 1;
			}
		}
		cout << "Threshold: " << threshold << " Error rate: " << errorRate / num << endl;
		file << "Threshold: " << threshold << " Error rate: " << errorRate / num << endl;
		threshold = threshold - 0.1;
	}

	file.close();

	// ouput the tree
	  file.open("tree.txt", ios::out);
	  file << treeNumber << endl;
	  file << depth << endl;
	  file << classNum << endl;
	  for (int t = 0;t < treeNumber;t++)
	  {
		travelTree(file, classNum, patchWidth, depth, 0, &roots[t]);
	  }
	  file.close();

}

void testTree2(fstream& file, mat& patches, vec& label, int classNum, int patchWidth, int depth)
{
	
	mat org;
	org.load("imagePatches_orig.mat", hdf5_binary);
	mat label_org;
	label_org.load("patchLabel_orig.mat", hdf5_binary);
	
	int treeNumber = 20;

	
	cout << "Test Total Tree Number: " << treeNumber << endl;
	file << "Test Total Tree Number: " << treeNumber << endl;

	TreeNode * roots = new TreeNode[treeNumber];

	int I = patches.n_cols;

	for (int t = 0; t < treeNumber; t++)
	{
		cout << "Tree number: " << t + 1 << endl;
		treebuilder(patches, label, classNum, patchWidth, depth, 0, &roots[t]);

		double errorRate = 0;
		for (int i = 0; i < I; i++)
		{
			vec data = patches.col(i);
			vec lambda = zeros<vec>(classNum);
			for (int tt = 0; tt <= t; tt++)
			{
				vec * tempLambda = testTree(data, classNum, patchWidth, &roots[tt]);
				lambda = lambda + (*tempLambda);
				delete tempLambda;
			}

			lambda = lambda / treeNumber;


			if ((lambda.index_max() + 1) != label(i))
			{
				errorRate++;
			}
		}

		cout << "Training Error rate: " << errorRate / I << endl;
		file << "Training Error rate: " << errorRate / I << endl;


		int num = org.n_cols;

		double threshold = 1;

		for (int k = 0; k < 9; k++)
		{
			errorRate = 0;
			for (int i = 0; i < num; i++)
			{
				vec data = org.col(i);
				vec lambda = zeros<vec>(classNum);
				for (int tt = 0; tt <= t; tt++)
				{
					vec * tempLambda = testTree(data, classNum, patchWidth, &roots[tt]);
					lambda = lambda + (*tempLambda);
					delete tempLambda;
				}

				lambda = lambda / treeNumber;


				uvec indices = sort_index(lambda, "descend");
				if (lambda(indices(1)) > threshold*lambda(indices(0)) || (((int)indices(0) + 1) != label_org(i)))
				{
					errorRate = errorRate + 1;
				}
			}
			cout << "Threshold: " << threshold << " Test Error rate: " << errorRate / num << endl;
			file << "Threshold: " << threshold << " Test  Error rate: " << errorRate / num << endl;
			threshold = threshold - 0.1;
		}

	}
	delete[]roots;
}


int main()
{
	srand(time(NULL));

	string imagePatchFilename = "imagePatches.mat";
	string patchLabelFilename = "patchLabel.mat";

	cout << "imagePatchFilename: " << imagePatchFilename << endl;
	cout << "patchLabelFilename: " <<  patchLabelFilename << endl;

	fstream file;
	
	// load training data
	mat patches;
	patches.load(imagePatchFilename, hdf5_binary);

	// load label of training data
	vec label;
	label.load(patchLabelFilename, hdf5_binary);

	// define number of classes
	int classNum =  50;

	// calculate width of patch
	int patchWidth = sqrt(patches.n_rows);

	// define depth of each tree
	int depth = 10;

	// generate and test classification tree
	testTree(patches, label, classNum, patchWidth, depth);

	// Test only
	/*
	file.open("testData_10.txt", ios::out);
	testTree2(file, patches, label, classNum, patchWidth, depth);
	file.close();

	depth = 5;
	file.open("testData_5.txt", ios::out);
	testTree2(file, patches, label, classNum, patchWidth, depth);
	file.close();
	*/

	return 0;
}