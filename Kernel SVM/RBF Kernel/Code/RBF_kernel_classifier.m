function predict_y = RBF_kernel_classifier(training_x, training_y, X, a, b, C, gamma)
    % INPUT : 
    % X   training data or test data
    % w   trained model
        
    % OUTPUT
    % predicted labels
m = size(X, 1);
predict_y = zeros(1, m);
for i = 1:m
    predict_y(i) = sign(a' * (training_y .* ...
    exp((-sum((training_x - X(i,:)) .^ 2, 2)) * gamma)) + b);
end
predict_y = predict_y';
end