clear;

fileID = fopen('tree.txt','r');

treeNumber = fscanf(fileID, '%d', 1);
depth = fscanf(fileID, '%d', 1);
classNum = fscanf(fileID, '%d', 1);

rootNodes = cell(treeNumber, 1);
for i=1:treeNumber
    display(i);
    rootNodes{i} = Node();
    reconstruct(fileID, classNum, rootNodes{i});
end

fclose(fileID);

imagePatches_orig = load('imagePatches_orig.mat');
imagePatches_orig = imagePatches_orig.imagePatches;
label = load('patchLabel_orig.mat');
label = label.label;

patchWidth = 33;
I = size(imagePatches_orig, 2);

lambdaMat = zeros(I, classNum);

error = 0;
errorLabel = zeros(I, 1);
for i = 1:I
    patch = reshape(imagePatches_orig(:, i), patchWidth, patchWidth);
    imshow(uint8(patch));
    drawnow
    
    lambda = zeros(classNum, 1);
    for t = 1:treeNumber
        tempLambda = testTree(patch, rootNodes{t});
        lambda = lambda + tempLambda;
    end
    lambda = lambda ./ treeNumber;
    
    lambdaMat(i, :) = lambda';
    
    [maxVal, maxLabel] = max(lambda);
    
    if maxLabel ~= label(i)
        errorLabel(i) = maxLabel;
        error = error + 1;
    end
end
error = error / I;

% original_img = imread('test.JPG');
% 
% gray_img = rgb2gray(original_img);
% 
% selected_points = load('selected_points.mat');
% selected_points = selected_points.selected_points;
% 
% I = size(selected_points, 2);
% patchWidth = 33;
% 
% for i = 1:I
%     
%     [patch] = getPatch(gray_img, selected_points(:,i), patchWidth);
%     
% end