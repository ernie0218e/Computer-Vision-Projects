clear;
cam = webcam(1);

for idx = 1:100
   % Acquire a single image.
   rgbImage = snapshot(cam);
   
   rgbImage = imresize(rgbImage, 0.8);

   % Convert RGB to grayscale.
   grayImage = rgb2gray(rgbImage);
   
   points = detectHarrisFeatures(grayImage);

   % Display the image.
   imshow(rgbImage);
   hold on;
   plot(points);
   drawnow
end

clear;