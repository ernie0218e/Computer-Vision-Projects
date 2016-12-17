clear;

data = load('imagePatches.mat');

patches = data.imagePatches;

patchWidth = sqrt(size(patches, 1));
D = patchWidth*patchWidth;
imageNum = size(patches, 2);

% classNum = data.K;
classNum =  200;

label = load('patchLabel.mat');
label = label.label;

% construct random tree
depth = 5;

root = cell(1, 1);
tree = cell((3^(depth+1) - 1)/2, 1);

subset = cell(3, 1);

[pt_dm1, pt_dm2, subset{1}, subset{2}, subset{3}] ...
    = optTree(patches, label, classNum, patchWidth);

node = cell(3, 1);
for i = 1:3
    node{3} = subset{i};
    tree{i} = node;
end

node{1} = pt_dm1;
node{2} = pt_dm2;
node{3} = [];
root{1} = node;

count = (3^(depth) - 1)/2;
for k = 1:count
    
    node = tree{k};
    
    [pt_dm1, pt_dm2, subset{1}, subset{2}, subset{3}] ...
    = optTree(node{3}{1}, node{3}{2}, classNum, patchWidth);
    
    node{1} = pt_dm1;
    node{2} = pt_dm2;
    
    for i = 1:3
        tree{3*k + i}{3} = subset{i};
    end
    
end

for k = (count+1):size(tree, 1)
    node = tree{k};
    lambda = zeros(classNum, 1);
   
    label = node{3}{2};
    
    I = size(label, 1);
    
    for c = 1:classNum
        lambda(c) = size(label(label == c), 1) / I;
    end
    
    tree{k} = lambda;
end