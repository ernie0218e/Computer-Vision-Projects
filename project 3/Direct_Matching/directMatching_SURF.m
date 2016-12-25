% Filename: directMatching_SURF
% Purpose: use matched SURF features and their locations to estimate
%          homography between reference picture and testing picture
clear;

original_img = imread('test.JPG');

% convert rgb image to gray scale
gray_img = rgb2gray(original_img);

% detect SURF feature
% use these features as reference
ref_points = detectSURFFeatures(gray_img);

% location of ref_points: I x 2
ref_points_loc = ref_points.Location;

% extrat feature descriptor
[ref_features, ref_validCorners] = extractFeatures(gray_img, ref_points);

% open webcam
cam = webcam(1);

rectPoints = [1 size(original_img, 2) size(original_img, 2) 1 1;...
            1 1 size(original_img, 1) size(original_img, 1) 1];

for idx = 1:500
    
    % Acquire a single image.
    frame = snapshot(cam);
    
    % convert rgb frame to gray scale
    gray_frame = rgb2gray(frame);

    % detect Surf feature
    points = detectSURFFeatures(gray_frame);

    % extrat feature descriptor
    [features, validCorners] = extractFeatures(gray_frame, points);
    
    
    % match features
    indexPairs = matchFeatures(ref_features, features, 'Unique', true);

    % extract matched point on both images;
    matchedPoints1 = ref_validCorners(indexPairs(:, 1));
    matchedPoints2 = validCorners(indexPairs(:, 2));

    % extract only the infomation of location
    matchedPoints1 = matchedPoints1.Location;
    matchedPoints2 = matchedPoints2.Location;
    
    % estimate homography
    [H, maxInliers] = findHomography(matchedPoints1, matchedPoints2);

    % Display the image.
    imshow([frame original_img]);
    % shift matchedPoints1 to draw points on the right side;
    matchedPoints1(:, 1) = matchedPoints1(:, 1) + size(frame, 2);
    hold on;
    % draw reference points
    scatter(ref_points_loc(:, 1) + size(frame, 2), ref_points_loc(:, 2), 'r');
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
    
    drawnow
end

% display(maxInliers);
