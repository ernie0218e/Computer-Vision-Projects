clear;
cam = webcam(1);

for idx = 1:100
   % Acquire a single image.
   rgbImage = snapshot(cam);
   
%    rgbImage = imresize(rgbImage, 0.8);

   % Convert RGB to grayscale.
   grayImage = rgb2gray(rgbImage);
   
   gray_img_f = single(grayImage);
%    points = detectHarrisFeatures(grayImage);
    [f, d] = vl_sift(gray_img_f) ;

    ref_points = f(1:2, :);

   % Display the image.
   imshow(rgbImage);
   hold on;
   scatter(ref_points(1,:), ref_points(2,:));
   drawnow
end

clear;