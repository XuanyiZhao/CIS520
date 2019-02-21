function predict_y = kernel_classifier(training_x, training_y, X, a, b, C, q)
    % INPUT : 
    % X   training data or test data
    % w   trained model
        
    % OUTPUT
    % predicted labels
    
% sv = find(a > 0.000001 * C);
% predict_y = sign(a(sv)' * (training_y(sv) .* ((training_x(sv,:) * X' + 1).^q)) + b);
predict_y = sign(a' * (training_y .* ((training_x * X' + 1).^q)) + b);
predict_y = predict_y';
end