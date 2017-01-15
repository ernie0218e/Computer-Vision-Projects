clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

loadFileName = 'data_PCA_200.mat';
    
% open and read the data
load(loadFileName);

lambdas = [1; 0.1; 0.01; 0.001; 0.0001; 0.00001; 0.000001; 0];

errorSR = zeros(500, size(lambdas, 1));
tr_errorSR = zeros(500, size(lambdas, 1));

for i = 1:size(lambdas, 1)

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
    eta = 0.01;
    w = label + 1;
    x = data;

    [phi, errorSR(:,i), tr_errorSR(:,i)] = multiLogisticBFGS(w, x, classNum, eta, lambdas(i), f_test_label, f_test_data);

end


figure(1);
for i = 1:size(errorSR, 2)
    plot(errorSR(errorSR(:,i) > 0,i));
    hold on;
end
title('Test error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('\lambda=1','\lambda=0.1','\lambda=0.01','\lambda=0.001',...
            '\lambda=0.0001','\lambda=0.00001','\lambda=0.000001','\lambda=0');
        
figure(2);
for i = 1:size(tr_errorSR, 2)
    plot(tr_errorSR(tr_errorSR(:,i) > 0,i));
    hold on;
end
title('Training error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('\lambda=1','\lambda=0.1','\lambda=0.01','\lambda=0.001',...
            '\lambda=0.0001','\lambda=0.00001','\lambda=0.000001','\lambda=0');
