
% Input: N - grid size
%        sigma - variance
function [f] = gaussian2d(N, sigma)
    
    % create a grid with size N and center the grid at origin
    [x, y]=meshgrid(round(-(N-1)/2):round((N-1)/2), round(-(N-1)/2):round((N-1)/2));
    f=exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
    f=f./sum(f(:));

end
