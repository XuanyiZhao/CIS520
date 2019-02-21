tic
% Load training data and test data
X_train_full = load('../Spam-Dataset/X_train.txt');
y_train_full = load('../Spam-Dataset/y_train.txt');
X_test_full = load('../Spam-Dataset/X_test.txt');
y_test_full = load('../Spam-Dataset/y_test.txt');

for i = 1:5
    X_train(:,:,i) = load(['../Spam-Dataset/CrossValidation/Fold',num2str(i),'/X.txt']);
    Y_train(:,:,i) = load(['../Spam-Dataset/CrossValidation/Fold',num2str(i),'/y.txt']);
end

initparams = [1, 5, 10, 15, 25, 50];
init_b1 = {};
init_b2 = {};
init_w1 = {};
init_w2 = {};
for p = 1:length(initparams)
    init_b1{p} = load(['../Spam-Dataset/InitParams/relu/',num2str(initparams(p)),'/b1.txt']);
    init_b2{p} = load(['../Spam-Dataset/InitParams/relu/',num2str(initparams(p)),'/b2.txt']);
    init_w1{p} = load(['../Spam-Dataset/InitParams/relu/',num2str(initparams(p)),'/W1.txt']);
    init_w2{p} = load(['../Spam-Dataset/InitParams/relu/',num2str(initparams(p)),'/W2.txt']);
end

training_error = [];
test_error = [];
cross_validation_error = [];
for j = 1:6
    average_error = [];
    for k = 1:5
        temp_x = X_train;
        temp_y = Y_train;
        X_test = temp_x(:,:,k);
        Y_test = temp_y(:,:,k);
        temp_x(:,:,k) = [];
        temp_y(:,:,k) = [];
        temp_train_x = [];
        temp_train_y = [];
        for s = 1:size(temp_x,3)
            temp_train_x = [temp_train_x; temp_x(:,:,s)];
            temp_train_y = [temp_train_y; temp_y(:,:,s)];
        end
        [w1, w2, b1, b2] = NN_ReLU_model(temp_train_x, temp_train_y, ...
            init_w1{j}, init_w2{j}, init_b1{j}, init_b2{j}, 8000, 0.1);
        predicted_y_test = NN_ReLU_classifier(X_test, w1, w2, b1, b2);
        error_test = classification_error(predicted_y_test, Y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
    
    [w1_2,w2_2, b1_2, b2_2] = NN_ReLU_model(X_train_full,y_train_full, ...
        init_w1{j}, init_w2{j}, init_b1{j}, init_b2{j}, 8000, 0.1);
    y_pred_train = NN_ReLU_classifier(X_train_full, w1_2, w2_2, b1_2, b2_2);
    error_train_full = classification_error(y_pred_train, y_train_full);
    training_error = [training_error, error_train_full];

    y_pred_test = NN_ReLU_classifier(X_test_full, w1_2, w2_2, b1_2, b2_2);
    error_test_full = classification_error(y_pred_test, y_test_full);
    test_error = [test_error, error_test_full];
end

figure
hold on
plot(initparams,training_error,'r',initparams,test_error,'b',...
    initparams,cross_validation_error)
xlabel('The number of hidden units');
ylabel('Error');
legend('Train error','Test error','Cross-validation error');
hold off

toc