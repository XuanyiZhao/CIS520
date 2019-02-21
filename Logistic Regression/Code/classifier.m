function predict_y = classifier(X, w)
    % INPUT : 
    % X   training data or test data
    % w   trained model
        
    % OUTPUT
    % predicted labels
predict_y = 1./(1 + exp(-X * w));
predict_y(predict_y > 1/2) = 1;
predict_y(predict_y <= 1/2) = -1;
end