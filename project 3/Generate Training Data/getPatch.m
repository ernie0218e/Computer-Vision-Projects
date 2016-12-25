%
% Filename: getPatch.m
% Purporse: get patch around the keypoint
% Input: img - entire image
%        point - the keypoint (2 x 1)
%        patchWidth - desired width of patch
% Output: patch - image patch centered at keypoint
%
function [patch] = getPatch(img, point, patchWidth)
    
    % calculate the leghth of half of patchWidth
    patchCenter = (patchWidth - 1) / 2;
        
    patch = 255.*rand(patchWidth, patchWidth);
    
    % if patch is inside the transformed image
    if (point(1, 1) - patchCenter >= 1) && (point(2, 1) - patchCenter >= 1)...
            && (point(1, 1) + patchCenter <= size(img, 2)) ...
            && (point(2, 1) + patchCenter <= size(img, 1))

        % get image patch around the keypoint
        patch = img((point(2, 1) - patchCenter):(point(2, 1) + patchCenter)...
                                       ,(point(1, 1) - patchCenter):(point(1, 1) + patchCenter));
    
    % else if patch is "partially" inside the transformed image
    else

        for r = (point(2, 1) - patchCenter):(point(2, 1) + patchCenter)
            for c = (point(1, 1) - patchCenter):(point(1, 1) + patchCenter)
                
                % if the point is inside the image
                % assign the value of that point to the patch
                if r >= 1 && r <= size(img, 1) && c >= 1 && c <= size(img, 2)
                    patch(r - (point(2, 1) - patchCenter) + 1, c - (point(1, 1) - patchCenter) + 1)...
                        = img(r, c);
                end
                
            end
        end
    end
end