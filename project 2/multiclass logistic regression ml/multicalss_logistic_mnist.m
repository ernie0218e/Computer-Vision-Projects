% Filename: multicalss_logistic_mnist.m
% Purpose: Train softmax regression model and test it
clear;

% open and read the data
load('data_PCA_200.mat');

% set up parameters
imageNum = 60000;
ratio = 1;
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

% add ones before data
data = [ones(imageNum, 1) data];

f_test_data = [ones(f_test_size, 1) f_test_data];


% number of class
classNum = 10;

% learning rate
eta = 0.5;

w = label + 1;
x = data;

% train model and get parameters phi
[phi] = multiLogistic(w, x, classNum, eta);

% compute test error
error = zeros(10, 1);
[lambda] = linearSoftMax(phi, f_test_data);
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

% compute training error
train_error = zeros(10, 1);
[lambda] = linearSoftMax(phi, data);
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