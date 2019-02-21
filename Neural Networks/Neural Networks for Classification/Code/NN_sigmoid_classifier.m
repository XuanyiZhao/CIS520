function predict_y = NN_sigmoid_classifier(X, w1, w2, b1, b2)
predict_y = w2 * sigmoid(w1 * X' + repmat(b1, 1, size(X, 1))) + b2;
predict_y = sigmoid(predict_y);
predict_y(predict_y > 1/2) = 1;
predict_y(predict_y <= 1/2) = -1; 
predict_y = predict_y';
end