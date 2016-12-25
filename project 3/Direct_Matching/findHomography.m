% Filename: findHomography.m
% Purpose: find homography between matchedPoints1 and matchedPoints2 using
%          RANSAC and Gauss-Newton method
% Input: matchedPoints1 - word (ref points) I x 2
%        matchedPoints2 - observed points I x 2
% Output: H - homography
%         maxInliers - number of Inliers
function [H, maxInliers] = findHomography(matchedPoints1, matchedPoints2)
    
    H = eye(3, 3);
    
    I = size(matchedPoints1, 1);
    maxInliers = 0;
    
    if I >= 4
    
        SubsetPoints = 4;
        
        threshold = 1;

        for t = 1:100
            % random chose 4 distinct matched points
            chose = randperm(I, SubsetPoints);

            % and the remain matched points are use to validate the homography
            otherMatchedPoints1 = matchedPoints1;
            otherMatchedPoints1(chose, :) = [];
            otherMatchedPoints2 = matchedPoints2;
            otherMatchedPoints2(chose, :) = [];
            
            % estimate Homography with four pairs of points
            tempH = estimateHomography(matchedPoints1(chose, :), matchedPoints2(chose, :));
            
            % Apply homography to the rest points
            newMatchedPoints1 = tempH*[otherMatchedPoints1';ones(1, I - SubsetPoints)];
            % Convert from homogeneous coordinates to Cartesian coordinate
            newMatchedPoints1(1, :) = newMatchedPoints1(1, :) ./ newMatchedPoints1(3, :);
            newMatchedPoints1(2, :) = newMatchedPoints1(2, :) ./ newMatchedPoints1(3, :);
            newMatchedPoints1(3, :) = [];
            
            % calculate number of inliers
            diff = sqrt((newMatchedPoints1(1,:) - otherMatchedPoints2(:, 1)').^2 ...
                + (newMatchedPoints1(2,:) - otherMatchedPoints2(:, 2)').^2);

            inliers = size(diff(diff < threshold), 2);
            if inliers > maxInliers
                maxInliers = inliers;
                H = tempH;
            end
        end
    end
end