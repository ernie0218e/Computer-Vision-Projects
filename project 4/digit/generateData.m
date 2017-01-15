%
% Filename: generateData.m
% Purpose: read and perprocess image data
%
clear;

% load original data from .mat file
dataList = load('lists.20.mat');
dataList = dataList.list;

% we only use data in the range of '0~9, A~Z'
totalSize = 36576;

% Make the 'label' of images
label = dataList.ALLlabels(1:totalSize, :);

% set size of new image
imgWidth = 16;

data = zeros(imgWidth^2, totalSize);
% for every data
for i = 1:totalSize
    
    % read image file
    imgPath = strcat('Fnt/', dataList.ALLnames(i, :), '.png');
    tempImg = imread(imgPath);
    
    % inverse the image
    tempImg = 255 - tempImg;
    
    % calculate the histogram
    histX = sum(tempImg, 1);
    histY = sum(tempImg, 2);
    
    % ------------- find the bounding box -----------------
    for t = 1:length(histX)
        if histX(t) ~= 0
            minX = t;
            break;
        end
    end

    for t = length(histX):-1:1
        if histX(t) ~= 0
            maxX = t;
            break;
        end
    end
    
    for t = 1:length(histY)
        if histY(t) ~= 0
            minY = t;
            break;
        end
    end

    for t = length(histY):-1:1
        if histY(t) ~= 0
            maxY = t;
            break;
        end
    end
    % ------------- find the bounding box -----------------
    
    % crop the image
    tempImg = imcrop(tempImg, [minX minY maxX-minX maxY-minY]);
    
    % resize the image
    % if width >= height
    if size(tempImg, 2) > size(tempImg, 1)
        
        % resize and keep the aspect ratio
        resizedImg = imresize(tempImg, [NaN imgWidth], 'bilinear');
        tempImg = zeros(imgWidth, imgWidth);
        
        height = size(resizedImg, 1);
        shift = round((imgWidth - height)/2);
        
        % move cropped image to the center of new image
        tempImg((1+shift):(shift+height), :) = resizedImg;
    else
        % resize and keep the aspect ratio
        resizedImg = imresize(tempImg, [imgWidth NaN], 'bilinear');
        tempImg = zeros(imgWidth, imgWidth);
        
        width = size(resizedImg, 2);
        shift = round((imgWidth - width)/2);
        
        % move cropped image to the center of new image
        tempImg(:, (1+shift):(shift+width)) = resizedImg;
    end
    
    % normalize
    tempImg = tempImg/max(max(tempImg));
    
    % vectorized
    data(:, i) = reshape(tempImg, imgWidth^2, 1);
    
end

save('data.mat', 'data');
save('label.mat', 'label');