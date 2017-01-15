% 
% Filename: fixPlateOrientation.m
% Purpose: fix the orientation of the license plate
% Input: plate - the image of the (candidate) license plate
%        grayImage - the intensity map of whole image
%        lpCoodinate - the up-left(Xmin, Ymin) coordinate of license plate
% Output: licensePlate - the image of the license plate after orientation
%                        correction
function [licensePlate] = fixPlateOrientation(plate, grayImage, lpCoordinate)
    
    % expand the region
    expandX = round(size(plate, 2) / 10);
    expandY = round(size(plate, 1) / 2);
    expandBound = [(lpCoordinate(1, 1) - expandX) ...
                    (lpCoordinate(1, 2) - expandY) ...
                    expandX * 12 ...
                    expandY * 4];
                
    % store the original coordinate of the license plate     
    originCoord = [(expandX - 1) (expandY - 1); ...
                   (size(plate, 2) + expandX + 1) (size(plate, 1) + expandY + 1)];
    
    % crop the expanded region
    expandRegion = imcrop(grayImage, expandBound);
    
    % use sobel operator to extract the info of edge
    BW = edge(expandRegion, 'sobel');
    
    % hough transformation
    [H,T,R] = hough(BW);
    
    % find 2 points which have the most curves intersecting on them.
    P = houghpeaks(H,2,'threshold',ceil(0.3*max(H(:))));
    
    % calculate correction of orientation
    % average the slope of the two longest line
    theta = (sum(T(P(:, 2))/2));
    % warning! the coordinate of the image is upside down.
    correctTheta = -cos(pi/180*theta)/sin(pi/180*theta)*180/pi;
    
    % Assume the license plate should be almost horizontal
    if abs(correctTheta) <= 30
    
        % rotate the image;
        correctImg = imrotate(expandRegion, correctTheta);

        % ---- rotate the points ----
        temp_x = originCoord(:, 1) - size(expandRegion, 2) / 2;
        temp_y = originCoord(:, 2) - size(expandRegion, 1) / 2;

        rotate = [cos(correctTheta*pi/180) -sin(correctTheta*pi/180);...
                    sin(correctTheta*pi/180) cos(correctTheta*pi/180)];

        points = rotate*[temp_x';temp_y'];

        points(1, :) = points(1, :) + size(correctImg, 2) / 2;
        points(2, :) = points(2, :) + size(correctImg, 1) / 2;
        points = round(points');
        % ---- rotate the points ----
        
        % crop the license plate after correction
        licensePlate = imcrop(correctImg, [points(1, 1) points(1, 2)...
                        points(2, 1)-points(1, 1) points(2, 2)-points(1, 2)]);
    else
        licensePlate = plate;
    end
    
    % code used to draw lines found by hough transform
%     lines = houghlines(BW,T,R,P,'FillGap',30,'MinLength',30);
%     figure, imshow(expandRegion), hold on
%     for k = 1:length(lines)
%        xy = [lines(k).point1; lines(k).point2];
%        plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%        % Plot beginnings and ends of lines
%        plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%        plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
%     end
end