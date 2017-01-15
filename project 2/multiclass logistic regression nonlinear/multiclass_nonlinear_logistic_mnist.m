% Filename: multiclass_nonlinear_logistic_mnist.m
% Purpose: Train nonlinear softmax regression model and test it
clear;

% open and read the data
load('data_PCA_200.mat');

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

% get the training data
data = (train_data(1:imageNum, 2:size(train_data, 2)));
label = train_data(1:imageNum, 1);

% get the testing data
f_test_size = size(test_data, 1);
f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
f_test_label = test_data(1:f_test_size, 1);

disp('End of read Data');

% number of class
classNum = 10;

% learning rate
eta = 0.0001;

% number of nonlinear function
K = 10;

w = label + 1;
x = data;

% Train nonlinear softmax regression model
[phi_zero, phi, zeta] = nonLinearMultiLogistic(w, x, classNum, K, eta);


error = zeros(10, 1);
% act = phi_zero
a = repmat(phi_zero, 1, f_test_size);
% act = \phi_0+\sum_{k=1}^{K}{\phi_katan(\zeta^{T}\mathbf{X})}
for k = 1:K
        a = a + repmat(phi(:, k), 1, f_test_size).*atan(squeeze(zeta(:, :, k))*f_test_data');
end
% softmax
den = sum(exp(a), 1);
lambda = exp(a) ./ repmat(den, classNum, 1);

for i = 1:f_test_size
    % find max value from the output of softmax function
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(lambda(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= f_test_label(i)
        error(f_test_label(i) + 1) = error(f_test_label(i) + 1) + 1;
    end
end

e = sum(error)/f_test_size;


train_error = zeros(10, 1);
% act = phi_zero
a = repmat(phi_zero, 1, imageNum);
% act = \phi_0+\sum_{k=1}^{K}{\phi_katan(\zeta^{T}\mathbf{X})}
for k = 1:K
        a = a + repmat(phi(:, k), 1, imageNum).*atan(squeeze(zeta(:, :, k))*data');
end
% softmax
den = sum(exp(a), 1);
lambda = exp(a) ./ repmat(den, classNum, 1);

for i = 1:imageNum
    % find max value from the output of softmax function
    % the label correspond to the max value is predicted result
    [maxVal, maxLabel] = max(lambda(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= label(i)
        train_error(label(i) + 1) = train_error(label(i) + 1) + 1;
    end
end

train_e = sum(train_error)/imageNum;