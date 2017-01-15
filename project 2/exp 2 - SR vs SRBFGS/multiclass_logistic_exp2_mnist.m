clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

loadFileName = 'data_PCA_200.mat';
    
% open and read the data
load(loadFileName);


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


classNum = 10;
ita = 0.01;
w = label + 1;
x = data;

[phi, errorSR, tr_errorSR] = multiLogistic(w, x, classNum, ita, f_test_label, f_test_data);


% train softmax regression (BFGS) model
classNum = 10;
ita = 0.01;
w = label + 1;
x = data;

[phi, errorSRBFGS, tr_errorSRBFGS] = multiLogisticBFGS(w, x, classNum, ita, f_test_label, f_test_data);

figure(1);
plot(errorSR);
hold on;
plot(errorSRBFGS(errorSRBFGS > 0));
hold on;
plot(tr_errorSR);
hold on;
plot(tr_errorSRBFGS(tr_errorSRBFGS > 0));
legend('test error of SR', 'test error of SR-BFGS', ...
           'training error of SR', 'training error of SR-BFGS');