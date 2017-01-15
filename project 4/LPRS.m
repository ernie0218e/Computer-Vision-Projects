%
% Filename: LPRS.m
% Purepose: The main function of the license plate recognition system
%
clear;

% load parameters of neural network
load('parameters.mat');

% define constant

% image width of each char for predicting
imgWidth = 16;

% aspect ratio of license plate
lparMax = 6;
lparMin = 2.3;

% minimum area of license plate
lpSizeMin = 15000;

% number of characters in a license plate
numOfchar = 7;

% read the image
rgbImage = imread('test\IMG_4342.JPG');

% get the original size of the image
imageWidth = size(rgbImage, 2);
imageHeight = size(rgbImage, 1);

% reduce size of the image
rgbImage = imcrop(rgbImage,...
   [round(imageWidth/6) round(imageHeight/6)...
   round(imageWidth*4/6) round(imageHeight*4/6)]);

% Convert RGB to intensity map
grayImage = (rgbImage(:,:,1)+rgbImage(:,:,2)+rgbImage(:,:,3))./3;

% histogram equalization
grayImage = histeq(grayImage);

% median filter
grayImage = medfilt2(grayImage);

% convert to double
grayImage = double(grayImage);

% edge detection using Sobel operator
H_Y = [-1 -2 -1;0 0 0;1 2 1];
H_X = [-1 0 1;-2 0 2;-1 0 1];
BW_Y = imfilter(grayImage, H_Y);
BW_X = imfilter(grayImage, H_X);

% compute edge-map
BW = sqrt(BW_Y.^2 + BW_X.^2);

% remove the value from image edges
BW(1, :) = 0;
BW(end, :) = 0;
BW(:, 1) = 0;
BW(:, end) = 0;

% compute average gradient
aveE = sum(sum(BW))/(size(BW,1)*size(BW,2));
% use the average of gradient as threshold
BW(BW <= aveE) = 0;

times = 3;
for t = 1:times
    % compute average gradient of remaining regions
    remainingSize = size(BW(BW ~= 0),1);
    aveE = sum(sum(BW))/remainingSize;
    % use the average of gradient as threshold
    BW(BW <= aveE) = 0;
end

% connect posible region of license plate
% horizontal

region = zeros(size(BW));
% define the threshold of maximum distance of two 'edge' pixels
% the pixel within these two pixel will consider as the same region
CD = size(BW, 2) / 50;
for r = 1:size(BW, 1)
  first_pt = 1;
  last_pt = 2;
  while last_pt <= size(BW, 2)
      if BW(r, last_pt) == 0
          last_pt = last_pt + 1;
      else
          if last_pt - first_pt <= CD
              region(r, first_pt:last_pt) = 1;
          end
          first_pt = last_pt;
          last_pt = last_pt + 1;
      end
  end
end


% find connected component
CC = bwconncomp(region);

% show the original rgb image
figure(4);
imshow(rgbImage);
hold on;

foundPlate = 0;
% select candidate number plate
maxE = 0;
maxRegions = 0;
for i = 1:CC.NumObjects

   % find bounding box
   rows = mod(CC.PixelIdxList{i}, size(BW, 1));
   cols = floor(CC.PixelIdxList{i} / size(BW, 1)) + 1;
    
   % the coners of bounding box
   maxY = max(rows);
   minY = min(rows);
   maxX = max(cols);
   minX = min(cols);

   % draw the bounding box
   rectangle('Position',[minX minY maxX-minX maxY-minY]...
       ,'EdgeColor','b', 'LineWidth',2);
   hold on;

   % extract interest region
   grayImage = uint8(grayImage);
   tempImg = imcrop(grayImage, [minX minY maxX-minX maxY-minY]);


   lpWidth = size(tempImg, 2);
   lpHeight = size(tempImg, 1);
   % calculate the aspect ratio of extracted region
   lpar = lpWidth / lpHeight;

   % if it could be a license plate
   if lpWidth*lpHeight >= lpSizeMin && ...
           lpar >= lparMin && lpar <= lparMax

       % draw the bounding box with different color
       rectangle('Position',[minX minY maxX-minX maxY-minY]...
       ,'EdgeColor','r', 'LineWidth',2);
   hold on;

        lpCoordinate = [minX minY];
        
        % Try to correct the orientation of the candidate plate first
        tempImg = fixPlateOrientation(tempImg, grayImage, lpCoordinate);
        
        % Detect the texture of the candidate plate
        [isPlate, regions] = detectPlate(tempImg, numOfchar, lparMax, lparMin);

        % if it is the license plate
        if isPlate && regions > maxRegions
            foundPlate = 1;
            licensePlate = imbinarize(tempImg);
            maxRegions = regions;
        end

   end
end

% if we've found the license plate
if foundPlate ~= 0

   % show the license plate
   figure(1);
   imshow(licensePlate);

   % segment the characters inside the license plate
   charImg = charSegmentation(licensePlate, numOfchar);

   % use neural network to classify the image
   characters = detectChar(charImg, imgWidth, parameters);

end