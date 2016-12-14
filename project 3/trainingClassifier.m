clear;

patches = load('imagePatches.mat');

% construct random tree
depth = 5;

tree = cell((3^(depth+1) - 1)/2, 1);