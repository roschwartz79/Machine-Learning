function [theta] = normalEqn(X_train, y_train)

theta = pinv(X_train' * X_train)*X_train'*y_train;

end