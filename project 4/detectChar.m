% 
% Filename: detectChar.m
% Purpose: classify the image of character using neural network
% Input: charImg - the image of the character
%        imgWidth - the size of image which is corresponded to the training
%                   data
%        parameters - parameters of neural network
% Output: characters - array of char
function [characters] = detectChar(charImg, imgWidth, parameters)
   
    % read the parameters
    W_xh1 = parameters{1};
    W_h1h2 = parameters{2};
    W_h2y = parameters{3};
    theta_h1 = parameters{4};
    theta_h2 = parameters{5};
    theta_y = parameters{6};
   
   % relationship between the 'label' and the 'characters'
   charRef = ['0';'1';'2';'3';'4';'5';'6';'7';'8';'9' ...
            ;'A';'B';'C';'D';'E';'F';'G';'H';'I';'J'...
            ;'K';'L';'M';'N';'O';'P';'Q';'R';'S';'T'...
            ;'U';'V';'W';'X';'Y';'Z'];
        
   data = zeros(imgWidth*imgWidth, size(charImg, 1));
   
   figure(11);
   for i = 1:size(charImg, 1)

       % get character
       tempImg = charImg{i};

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
       
       % vectorized the image
       data(:, i) = reshape(tempImg, imgWidth*imgWidth, 1);
       
       % normalize
       data(:, i) = data(:, i)./max(data(:, i));
       
       % show each image
       subplot(1, size(charImg, 1), i), imshow(tempImg);
   end
   
   characters = char(zeros(size(charImg, 1), 1));

   % Use neural network for prediction
   Y = MLP_Predict(data', W_xh1, W_h1h2, W_h2y, theta_h1, theta_h2, theta_y);
   for i = 1:size(data, 2)
        % find max value from the output of neural network
        % the label correspond to the max value is predicted result
        [maxVal, maxLabel] = max(Y(:, i));
        % display the result
        disp(charRef(maxLabel, :));
        characters(i) = charRef(maxLabel, :);
   end

end