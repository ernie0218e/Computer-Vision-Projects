patches = cell(K, 1);

for j = 1:K
%         % add random shift
%         shift = randi(4, 1, 2)-2;
%         selected_points(j, :) = shift + selected_points(j, :);
        tempPatch = getPatch(gray_img, round(selected_points(j, :)'), patchWidth);
        
%         tempPatch = imgaussfilt(imnoise(tempPatch, 'gaussian', 0, 0.01));
        
        % calculate gradient of image patch
        [gradPatchMag, gradPatchDir] = imgradient(tempPatch);
        
        weighted = gradPatchMag .* gaussianKernel2d;
        % vectorized
        gradPatchDir = reshape(gradPatchDir, D, 1);
        weighted = reshape(weighted, D, 1);
        
        % calculate histogram of gradient
        nbins = 36;
%         [N, edges] = histcounts(gradPatchDir, nbins);
        [N, vinterval] = histwc(gradPatchDir, -180, 180, weighted, nbins);
        % find orientation of the image patch based on maximum value of
        % gradient
        [maxValue, maxIndex]=max(N);
        patchTheta = 2*pi/nbins*maxIndex - pi;
        
%         temp = (patchTheta/(2*pi)*360);
        
        % normalize orientation
%         correctImage = imrotate(gray_img, -(patchTheta/(2*pi)*360));
        correctImage = rotateAround(gray_img, selected_points(j, 2), selected_points(j, 1), -(patchTheta/(2*pi)*360));
        
%         % ---- rotate the point ----
%         temp_x = selected_points(j, 1) - size(gray_img, 2) / 2;
%         temp_y = selected_points(j, 2) - size(gray_img, 1) / 2;
%         
%         R_temp = [cos(patchTheta) -sin(patchTheta);sin(patchTheta) cos(patchTheta)];
%         
%         correctPoint = R_temp*[temp_x;temp_y];
% 
%         correctPoint(1, 1) = correctPoint(1, 1) + size(correctImage, 2) / 2;
%         correctPoint(2, 1) = correctPoint(2, 1) + size(correctImage, 1) / 2;
%         correctPoint = round(correctPoint);
        % --------------------------
        
        patches{j} = getPatch(correctImage, round(selected_points(j, :)'), patchWidth);
        patches{j} = imgaussfilt(patches{j});
end
    
D = patchWidth*patchWidth;
imageNum = K;
imagePatches = zeros(D, imageNum);
for i = 1:imageNum
    imagePatches(:, i) = reshape(patches{i}, D, 1);
end
imagePatches = uint8(imagePatches);
save('imagePatches_orig.mat', 'imagePatches', '-v7.3');