clear;

data = load('imagePatches.mat');

D = size(data.patches{1}, 1)*size(data.patches{1}, 2);
imageNum = size(data.patches, 1);

patches = zeros(D, imageNum);
for i = 1:imageNum
    patches(:, i) = reshape(data.patches{i}, D, 1);
end

% classNum = data.K;
classNum =  200;

label = repmat((1:classNum)', round(imageNum / classNum), 1);

% construct random tree
depth = 5;

tree = cell((3^(depth+1) - 1)/2, 1);



count = (3^(depth) - 1)/2 - 1;
for k = 1:count
    
    
end