% Filename: testingImageGenerator.m
% Purporse: generate testing images with random affine transformation
clear;

% read original image
original_img = imread('test.JPG');

% convert rgb image to gray scale
gray_img = rgb2gray(original_img);

% load stable key points
selected_points = load('selected_points.mat');
selected_points = selected_points.selected_points;

% how many key points in one image
K = size(selected_points, 1);

% how many affine transformations will be used
times = 100;

patches = cell(K*times, 1);
patchWidth = 33;
D = patchWidth*patchWidth;

% generate normalized gaussian kernel
gaussianKernel2d = gaussian2d(patchWidth, (patchWidth -  1)/4);
gaussianKernel2d = gaussianKernel2d ./ max(max(gaussianKernel2d));

for i = 1:times
    
    % generate parameters for random affine transformation
    phi = 2*pi*rand(1, 1)-pi;
    theta = 2/3*pi*rand(1, 1)-pi/3;  
    S = diag(1.2*rand(3, 1)+0.4);
    S(3, 3) = 1;

    R_phi = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
    R_theta = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    
    % calculate affine transformation matrix
    A = R_theta*inv(R_phi)*S*R_phi;
    
    tform = affine2d(A');
    
    % apply affine transformation to gray image
    tempImage = imwarp(gray_img, tform);
    
    % ---- rotate key points ----
    temp_x = selected_points(:, 1) - size(gray_img, 2) / 2;
    temp_y = selected_points(:, 2) - size(gray_img, 1) / 2;
    
    points = A*[temp_x';temp_y';ones(1, K)];
    
    points(1, :) = points(1, :) + size(tempImage, 2) / 2;
    points(2, :) = points(2, :) + size(tempImage, 1) / 2;
    points(3, :) = [];
    points = round(points);
    % --------------------------
    
    % for each keypoint
    for j = 1:K
        
        % get patch around the key point
        tempPatch = getPatch(tempImage, points(:, j), patchWidth);
        
        % calculate gradient of image patch
        [gradPatchMag, gradPatchDir] = imgradient(tempPatch);
        
        % calculate weight
        weighted = gradPatchMag .* gaussianKernel2d;
        % vectorized
        gradPatchDir = reshape(gradPatchDir, D, 1);
        weighted = reshape(weighted, D, 1);
        
        % calculate histogram of gradient based on weight
        nbins = 36;
        [N, vinterval] = histwc(gradPatchDir, -180, 180, weighted, nbins);
        % find orientation of the image patch based on maximum value of
        % gradient
        [maxValue, maxIndex]=max(N);
        patchTheta = 2*pi/nbins*maxIndex - pi;
        
        % normalize orientation (rotate around the key point)
        correctImage = rotateAround(tempImage, points(2, j), points(1, j), -(patchTheta/(2*pi)*360));
        
        % get patch from correct image
        patches{K*(i-1)+j} = getPatch(correctImage, points(:, j), patchWidth);
        
        % add noise and apply gaussian filter to the patch
        patches{K*(i-1)+j} = imgaussfilt(imnoise(patches{K*(i-1)+j}, 'gaussian', 0, 0.01));
        
    end
end

% vectorized patches
D = patchWidth*patchWidth;
imageNum = K*times;
imagePatches = zeros(D, imageNum);
for i = 1:imageNum
    imagePatches(:, i) = reshape(patches{i}, D, 1);
end

% output image patches for testing
save('imagePatches_orig.mat', 'imagePatches', '-v7.3');

% generate patche labels for testing
label = repmat((1:K)', round(imageNum / K), 1);
save('patchLabel_orig.mat', 'label', '-v7.3');