function predict_y = classifier(X, w, b)
    % INPUT : 
    % X   training data or test data
    % w   trained model
        
    % OUTPUT
    % predicted labels
predict_y = sign(X * w' + b);
end