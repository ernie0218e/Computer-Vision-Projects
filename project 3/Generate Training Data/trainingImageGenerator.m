% Filename: trainingImageGenerator.m
% Purporse: generate training images with random affine transformation
clear;

% read original image
original_img = imread('test.JPG');

% convert rgb image to gray scale
gray_img = rgb2gray(original_img);

% detect harris features of original gray scale image
% use these features as reference
ref_points = detectHarrisFeatures(gray_img);

ref_points = ref_points.Location;

% size of generated view set
times = 600;

% store affine transformation matrix
transformations = cell(times, 1);

ref_points_index = zeros(size(ref_points, 1), 1);

for t = 1:times
    
    % generate parameters for random affine transformation
    phi = 2*pi*rand(1, 1)-pi;
    theta = 2/3*pi*rand(1, 1)-pi/3;
    S = diag(1.2*rand(3, 1)+0.4);
    
    S(3, 3) = 1;

    R_phi = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
    R_theta = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    
    % calculate affine transformation matrix
    A = R_theta*inv(R_phi)*S*R_phi;
    
    if t == 1
        A = eye(3, 3);
    end
    
    % store trasformation matrix
    transformations{t} = A;
    tform = affine2d(A');
    
    % apply affine transformation to gray image
    J = imwarp(gray_img, tform);
    % add noise  to transformed gray image
    J = imnoise(J, 'gaussian', 0, 0.01);
        
    % detect harris feature
    points = detectHarrisFeatures(J);
    
    % ---- rotate the points ----
    temp_x = points.Location(:,1);
    temp_y = points.Location(:,2);
    
    temp_x = temp_x - size(J, 2) / 2;
    temp_y = temp_y - size(J, 1) / 2;
    
    telta_points = inv(A) *...
        [temp_x temp_y ones(size(points.Location, 1), 1)]';
    
    telta_points(1, :) = telta_points(1, :) + size(gray_img, 2) / 2;
    telta_points(2, :) = telta_points(2, :) + size(gray_img, 1) / 2;
    telta_points(3, :) = [];
    % --------------------------
    
    % check whether the  keypoints are matched with reference key points
    for i = 1:size(ref_points, 1)
        for j = 1:size(telta_points, 2)
            if sqrt((ref_points(i, 1) - telta_points(1, j))^2 ...
                    +(ref_points(i, 2) - telta_points(2, j))^2) < 2
                ref_points_index(i) = ref_points_index(i) + 1;
            end
        end
    end
    
end

% find stable key points based on the number of occurrences
[sorted_ref_points_index, sortingIndices] = sort(ref_points_index, 'descend');

K = 50;

% select K key points
selected_points = ref_points(sortingIndices(1:K), :);

patches = cell(K*times, 1);

% set patch width
patchWidth = 33;
D = patchWidth*patchWidth;

% generate normalized gaussian kernel
gaussianKernel2d = gaussian2d(patchWidth, (patchWidth -  1)/4);
gaussianKernel2d = gaussianKernel2d ./ max(max(gaussianKernel2d));

for i = 1:times
    A = transformations{i};
    
    tform = affine2d(A');
    % apply affine transformation to gray image
    tempImage = imwarp(gray_img, tform);
    
    % ---- rotate the points ----
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
        
        % calculate histogram of gradient based on the weight of magnitude
        nbins = 36;
        [N, vinterval] = histwc(gradPatchDir, -180, 180, weighted, nbins);

        % find orientation of the image patch based on maximum value of
        % gradient
        [maxValue, maxIndex]=max(N);
        patchTheta = 2*pi/nbins*maxIndex - pi;
        
        % normalize orientation
        % rotate around the keypoint
        correctImage = rotateAround(tempImage, points(2, j), points(1, j), -(patchTheta/(2*pi)*360));
        
        % get patch from correct image
        patches{K*(i-1)+j} = getPatch(correctImage, points(:, j), patchWidth);
        
        % add noise and apply gaussian filter to the patch
        patches{K*(i-1)+j} = imgaussfilt(imnoise(patches{K*(i-1)+j}, 'gaussian', 0, 0.01)); 
    end
end

% figure(2);
% subplot(2,2,1), imshow(patches{K})
% subplot(2,2,2), imshow(patches{2*K})
% subplot(2,2,3), imshow(patches{3*K})
% subplot(2,2,4), imshow(patches{4*K})
% figure(2);
% for i = 1:16
%     subplot(4, 4, i), imshow(patches{K*(i-1) + 1})
% end

% figure(1);
% imshow(original_img); hold on;
% scatter(selected_points(:, 1), selected_points(:, 2));
% 
% figure(2);
% imshow(original_img); hold on;
% scatter(ref_points(:, 1), ref_points(:, 2));

% vectorized patches
D = patchWidth*patchWidth;
imageNum = K*times;
imagePatches = zeros(D, imageNum);
for i = 1:imageNum
    imagePatches(:, i) = reshape(patches{i}, D, 1);
end

% output image patches for training
save('imagePatches.mat', 'imagePatches', '-v7.3');

% generate patche labels for training
label = repmat((1:K)', round(imageNum / K), 1);

save('patchLabel.mat', 'label', '-v7.3');

% save the location of stable key points
save('selected_points.mat', 'selected_points', '-v7.3');
