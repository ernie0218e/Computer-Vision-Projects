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
times = 500;

% store affine transformation matrix
transformations = cell(times, 1);

ref_points_index = zeros(size(ref_points, 1), 1);

for t = 1:times
    phi = 2*pi*rand(1, 1)-pi;
    theta = 2*pi*rand(1, 1)-pi;
    S = diag(1.6*rand(3, 1)+0.2);
    S(3, 3) = 1;

    R_phi = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
    R_theta = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];

    A = R_theta*inv(R_phi)*S*R_phi;
    
    transformations{t} = A;

    tform = affine2d(A');
    
    J = imwarp(gray_img, tform);
    J = imnoise(J, 'gaussian', 0, 0.01);

    points = detectHarrisFeatures(J);
    
    temp_x = points.Location(:,1);
    temp_y = points.Location(:,2);
    
    temp_x = temp_x - size(J, 2) / 2;
    temp_y = temp_y - size(J, 1) / 2;
    
    telta_points = inv(A) *...
        [temp_x temp_y ones(size(points.Location, 1), 1)]';
    
    telta_points(1, :) = telta_points(1, :) + size(gray_img, 2) / 2;
    telta_points(2, :) = telta_points(2, :) + size(gray_img, 1) / 2;
    telta_points(3, :) = [];
    
    for i = 1:size(ref_points, 1)
        for j = 1:size(telta_points, 2)
            if sqrt((ref_points(i, 1) - telta_points(1, j))^2 ...
                    +(ref_points(i, 2) - telta_points(2, j))^2) < 2
                ref_points_index(i) = ref_points_index(i) + 1;
            end
        end
    end
    
end

[sorted_ref_points_index, sortingIndices] = sort(ref_points_index, 'descend');

K = 200;

selected_points = ref_points(sortingIndices(1:K), :);


figure(1);
imshow(original_img); hold on;
scatter(selected_points(:, 1), selected_points(:, 2));

figure(2);
imshow(original_img); hold on;
scatter(ref_points(:, 1), ref_points(:, 2));

figure(3);
imshow(J); hold on;
plot(points);