num_train = 721;
num_test = 309;
X_train_100 = load('../Data-set-2/Train-subsets/X_train_100%.txt');
X_train_100(:,9) = ones(num_train,1);
Y_train_100 = load('../Data-set-2/Train-subsets/y_train_100%.txt');
X_test100 = load('../Data-set-2/X_test.txt');
X_test100(:,9) = ones(num_test,1);
Y_test100 = load('../Data-set-2/y_test.txt');

train_error_100 = [];
test_error_100 = [];
value_of_lambda = [];
lambda_vector = [0.1, 1, 10, 100, 500, 1000];
I = eye(8);
I(:,9)=zeros(8,1);
I(9,:)=zeros(9,1);
for l = 1:6
    lambda = lambda_vector(l);
    value_of_lambda = [value_of_lambda, lambda];
    w2_b2 = (X_train_100' * X_train_100 + lambda * num_train * I)^-1 ...
        * X_train_100' * Y_train_100;
    predicted_y_train100 = X_train_100 * w2_b2;
    predicted_y_test100 = X_test100 * w2_b2;
    train_error_100 = [train_error_100, mean_squared_error(Y_train_100, predicted_y_train100)];
    test_error_100 = [test_error_100, mean_squared_error(Y_test100, predicted_y_test100)];
end

X_train(:,:,1) = [load(['../Data-set-2/Cross-validation/Fold1/X_100%.txt']), ones(145,1)];
Y_train(:,:,1) = load(['../Data-set-2/Cross-validation/Fold1/y_100%.txt']);

for i = 2:5
    X_train(:,:,i) = [[load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/X_100%.txt']); zeros(1,8)], ones(145,1)];
    Y_train(:,:,i) = [load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/y_100%.txt']); zeros(1,1)];
end

cross_validation_error = [];
for j = 1:6
    average_error = [];
    lambda = lambda_vector(j);
    for k = 1:5
        temp_x = X_train;
        temp_y = Y_train;
        if (sum(temp_x(end,:,k)) ~= 1) && (temp_y(end,:,k) ~= 0)
            X_test = temp_x(:,:,k);
            Y_test = temp_y(:,:,k);
        elseif (sum(temp_x(end,:,k)) == 1) || (temp_y(end,:,k) == 0)
            X_test = temp_x(1:end-1,:,k);
            Y_test = temp_y(1:end-1,:,k);
        end
        temp_x(:,:,k) = [];
        temp_y(:,:,k) = [];
        temp_train_x = [];
        temp_train_y = [];
        for s = 1:size(temp_x,3)
            if (sum(temp_x(end,:,s)) ~= 1) && (temp_y(end,:,s) ~= 0)
                temp_train_x = [temp_train_x; temp_x(:,:,s)];
                temp_train_y = [temp_train_y; temp_y(:,:,s)];
            elseif (sum(temp_x(end,:,s)) == 1) || (temp_y(end,:,s) == 0)
                temp_train_x = [temp_train_x; temp_x(1:end-1,:,s)];
                temp_train_y = [temp_train_y; temp_y(1:end-1,:,s)]; 
            end
        end
        w_b = (temp_train_x' * temp_train_x + lambda * num_train * I)^-1 ...
        * temp_train_x' * temp_train_y;
        predicted_y_test = X_test * w_b;
        error_test = mean_squared_error(Y_test,predicted_y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
end

figure
hold on
plot([-1, 0, 1, 2, log10(500), 3],train_error_100,'r',[-1, 0, 1, 2, log10(500), 3],test_error_100,'b',...
    [-1, 0, 1, 2, log10(500), 3],cross_validation_error)
xlabel('Log10(Value of lambda)');
ylabel('Error');
legend('Train error','Test error','Cross-validation error');
hold off