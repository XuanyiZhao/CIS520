function predict_y = NN_ReLU_classifier(X, w1, w2, b1, b2)
predict_y = w2 * ReLU(w1 * X' + repmat(b1, 1, size(X, 1))) + b2;
predict_y = predict_y';
end