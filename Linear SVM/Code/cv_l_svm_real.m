% Load the 5-fold cross-validation data
for i = 1:5
    X_train(:,:,i) = load(['../Spam-Dataset/CrossValidation/Fold',num2str(i),'/X.txt']);
    Y_train(:,:,i) = load(['../Spam-Dataset/CrossValidation/Fold',num2str(i),'/y.txt']);
end

% Load training data and test data
X_train_full = load('../Spam-Dataset/X_train.txt');
y_train_full = load('../Spam-Dataset/y_train.txt');
X_test_full = load('../Spam-Dataset/X_test.txt');
y_test_full = load('../Spam-Dataset/y_test.txt');

exp_c = [-4, -3, -2, -1, 0, 1, 2];
training_error = [];
test_error = [];
cross_validation_error = [];

for i = 1:length(exp_c)
    average_error = [];
    para_c = 10^exp_c(i);
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
        [w,b] = linear_svm_model(temp_train_x,temp_train_y,para_c);
        y_predicted_test = classifier(X_test, w, b);
        error_test = classification_error(y_predicted_test,Y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
    
    [w_1,b_1] = linear_svm_model(X_train_full,y_train_full,para_c);
    error_train_full = classification_error(classifier(X_train_full, w_1, b_1), y_train_full);
    training_error = [training_error, error_train_full];
    
    error_test_full = classification_error(classifier(X_test_full, w_1, b_1), y_test_full);
    test_error = [test_error, error_test_full];
end

figure
hold on
plot(exp_c,training_error,'r',exp_c,test_error,'b',exp_c,cross_validation_error,'c')
xlabel('Log10(C)');
ylabel('Error');
legend('Train error','Test error','Cross-validation error');
hold off
