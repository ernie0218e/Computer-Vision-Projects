% Filename: matching.m
% Purpose: use classification tree to match keypoints in reference picture
%          and testing picture and then estimate homography between two
%          images
clear;

% read the info of tree
fileID = fopen('tree.txt','r');

treeNumber = fscanf(fileID, '%d', 1);
depth = fscanf(fileID, '%d', 1);
classNum = fscanf(fileID, '%d', 1);

% reconstruct the tree
rootNodes = cell(treeNumber, 1);
for i=1:treeNumber
    display(i);
    rootNodes{i} = Node();
    reconstruct(fileID, classNum, rootNodes{i});
end

fclose(fileID);

patchWidth = 33;
D = patchWidth*patchWidth;
threshold = 0.6;

% generate normalized gaussian kernel
gaussianKernel2d = gaussian2d(patchWidth, (patchWidth -  1)/4);
gaussianKernel2d = gaussianKernel2d ./ max(max(gaussianKernel2d));

% load stable key points
ref_points = load('selected_points.mat');
ref_points = ref_points.selected_points;

% open webcam
cam = webcam(1);

original_img = imread('test.JPG');

% rectangle around the reference image
rectPoints = [1 size(original_img, 2) size(original_img, 2) 1 1;...
            1 1 size(original_img, 1) size(original_img, 1) 1];

for idx = 1:500
    % Acquire a single image.
    frame = snapshot(cam);

    % convert rgb frame to gray scale
    gray_frame = rgb2gray(frame);
    
    % detect Harris feature and get the locations of key points
    selected_points = detectHarrisFeatures(gray_frame);
    selected_points = selected_points.Location;

    I = size(selected_points, 1);

    matched_idx = zeros(I, 3);
    error = 0;
    
    % for each key point
    for i = 1:I
        
        % get patch around the key point
        tempPatch = getPatch(gray_frame, round(selected_points(i, :)'), patchWidth);

        % calculate gradient of image patch
        [gradPatchMag, gradPatchDir] = imgradient(tempPatch);
        
        % calculate weight
        weighted = gradPatchMag .* gaussianKernel2d;
        % vectorized
        gradPatchDir = reshape(gradPatchDir, D, 1);
        weighted = reshape(weighted, D, 1);

        % calculate weighted histogram of gradient
        nbins = 36;
        [N, vinterval] = histwc(gradPatchDir, -180, 180, weighted, nbins);
        
        % find orientation of the image patch based on maximum value of
        % gradient
        [maxValue, maxIndex]  = max(N);
        patchTheta = 2*pi/nbins*maxIndex - pi;
        % normalize orientation
        correctImage = ...
            rotateAround(gray_frame, selected_points(i, 2), selected_points(i, 1),...
            -(patchTheta/(2*pi)*360));

        patch = getPatch(correctImage, round(selected_points(i, :)'), patchWidth);
        patch = double(patch);
        patch = imgaussfilt(patch);

        % test patch
        lambda = zeros(classNum, 1);
        for t = 1:treeNumber
            tempLambda = testTree(patch, rootNodes{t});
            % sum the probability of each class from different tree
            lambda = lambda + tempLambda;
        end
        lambda = lambda ./ treeNumber;
        
        % sort probability of each class
        [sorted_val, sorted_idx] = sort(lambda, 1, 'descend');
        
        % if (maximum value*threshold < second large value)
        % --> (view that patch as background)
        matched_idx(i, 1) = i;
        if (sorted_val(1, 1)*threshold < sorted_val(2, 1))
            matched_idx(i, 2) = -1;
            matched_idx(i, 3) = 0;
        else
            matched_idx(i, 2) = sorted_idx(1, 1);
            matched_idx(i, 3) = sorted_val(1, 1);
        end
    end
    
    % make matched points unique
    matched_idx = sortrows(matched_idx, -3);
    [C,IA,IC] = unique(matched_idx(:, 2),'first');
    matched_idx = matched_idx(IA, :);
    if matched_idx(1, 2) == -1
       matched_idx(1, :) = [];
    end

    matchedPoints1 = ref_points(matched_idx(:, 2), :);
    matchedPoints2 = selected_points(matched_idx(:, 1), :);
    
    % estimate homography
    [H, maxInliers] = findHomography(matchedPoints1, matchedPoints2);
    
    % Display the image.
    imshow([frame original_img]);
    % shift matchedPoints1 to draw points on the right side;
    matchedPoints1(:, 1) = matchedPoints1(:, 1) + size(frame, 2);
    hold on;
    % draw reference points
    scatter(ref_points(:, 1) + size(frame, 2), ref_points(:, 2), 'r');
    hold on;
    % draw matched points on reference image
    scatter(matchedPoints1(:, 1), matchedPoints1(:, 2), 'b');
    hold on;
    % draw matched points on input image
    scatter(matchedPoints2(:, 1), matchedPoints2(:, 2), 'b');
    hold on;
    % draw lines between matched points
    pts_x = [matchedPoints1(:, 1) matchedPoints2(:, 1)]';
    pts_y = [matchedPoints1(:, 2) matchedPoints2(:, 2)]';
    plot(pts_x, pts_y, 'g');
    if maxInliers ~= 0
        hold on;
        newRectPoints = H*[rectPoints;ones(1, 5)];
        newRectPoints(1, :) = newRectPoints(1, :) ./ newRectPoints(3, :);
        newRectPoints(2, :) = newRectPoints(2, :) ./ newRectPoints(3, :);
        newRectPoints(3, :) = [];
        plot(newRectPoints(1, :),newRectPoints(2,:),'Color','r','LineWidth',4);
        display(maxInliers);
    end
    hold on;
    scatter(matchedPoints2(:, 1), matchedPoints2(:, 2));
    drawnow
    
end

