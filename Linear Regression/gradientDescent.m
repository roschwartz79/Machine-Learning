function [theta, cost] = gradientDescent(X_train, y_train, alpha, iters)

% X_train -> m x n FEATURE matrix with m SAMPLES and n FEATURE DIMENSIONS
% y_train -> is m x 1 with LABELS for training set
% alpha -> the learning rate to use in WEIGHT UPDATE
% iters -> the num of ITERATIONS to run g.d. for

% How many samples and features do we have
m = length(y_train);
cols = size(X_train,2);

% Data matrix to be returned
cost = zeros(iters,1);

% Init theta0 and theta1 to random nums, 1 and 0
theta = zeros(cols,1);

for i = 1:iters
    % What is our hypotheses?
    % h_x = X_train(:,1)* theta(1) + X_train(:,2) * theta(2)...
    h_x = X_train * theta;
    
    % Iterating over the different theta values/features
    for j = 1:cols
        % Compute Gradient Descent into a temp variable
        %temp_theta0 = theta(1) - (alpha*(1/m)*sum(h_x(:) - y_train(:,1)));
        %temp_theta1 = theta(2) - (alpha*(1/m)*sum((h_x(:) - y_train(:,1)).*(X_train(:,2))));
        temp_theta = theta(j) - (alpha*(1/m)*sum((h_x(:) - y_train(:)).*(X_train(:,j))));
        
        % Set the theta values to the temp values
        %theta(1) = temp_theta0;
        %theta(2) = temp_theta1;
        theta(j) = temp_theta;
    end
    % compute the cost at each iteration & put into cost array
    cost(i) = computeCost(X_train, y_train, theta);
end 

end
