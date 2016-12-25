run('E:/VLfeat/vlfeat-0.9.20/toolbox/vl_setup.m');
clear;
original_img = imread('test.JPG');

% convert rgb image to gray scale
gray_img = rgb2gray(original_img);

% detect SIFT feature
% use these features as reference
gray_img_f = single(gray_img);
[reff, refd] = vl_sift(gray_img_f);

ref_points = reff(1:2, :)';

cam = webcam(1);

rectPoints = [1 size(original_img, 2) size(original_img, 2) 1 1;...
            1 1 size(original_img, 1) size(original_img, 1) 1];

for idx = 1:500
    
    % Acquire a single image.
    frame = snapshot(cam);
    
    
    % convert rgb frame to gray scale
    gray_frame = rgb2gray(frame);
    
    % detect SIFT feature
    gray_frame_f = single(gray_frame);
    [f, d] = vl_sift(gray_frame_f);
    points = f(1:2, :)';
    
    % match features
    [matches, scores] = vl_ubcmatch(refd, d);
    

    % extract matched point on both images;
    matchedPoints1 = ref_points(matches(1, :), :);
    matchedPoints2 = points(matches(2, :), :);

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
    drawnow
end
