% Problem Set 2, ECE 1395
% Written by Rob Schwartz
clc
clear all
close all

% Separate the file by spaces into a matrix
fileID = fopen('input/hw2_data1.txt');
rawData = textscan(fileID,'%f %f','Delimiter',',');
fclose(fileID);

% Set up the data 
X_raw = cell2mat(rawData(:,1));
X = ones(length(X_raw), 2);
X(:,2) = X_raw;
y = cell2mat(rawData(:,2));

% Plot the data
figure(1);
scatter(X_raw,y,'x','red');
xlabel("Population of City in 10,000s");
ylabel("Profit in 10,000s");

% set up the zero model parameter theta vector
theta = [0;0];

J = computeCost(X, y, theta);
fprintf("The cost with zero model parameters: %s", num2str(J));

% Compute gradient descent alpha = .1, iters = 1500
[theta, cost] = gradientDescent(X, y, .01, 1500);

% Cost after running gradient descent
J = computeCost(X, y, theta);

fprintf("\n\nTheta 0 is equal to %s and Theta 1 is equal to %s", num2str(theta(1)), num2str(theta(2)));

% Plot the cost vector
figure(2);
plot(cost);
title("Cost vector over each iteration");
xlabel("Iteration");
ylabel("Cost");

% Use the model params to make predictions on profits in cities of 35k and
% 70k
prediction_35k_gd = theta(1) + theta(2)*35000;
prediction_70k_gd = theta(1) + theta(2)*70000;

fprintf("\n\nThe prediction for profits in the city with 35000: %s",num2str(prediction_35k_gd));
fprintf("\nThe prediction for profits in the city with 70000: %s",num2str(prediction_70k_gd));

% Use the normal equation
theta = normalEqn(X, y);
fprintf("\n\nTheta 0 is equal to %s and Theta 1 is equal to %s", num2str(theta(1)), num2str(theta(2)));

% Use the model params to make predictions on profits in cities of 35k and
% 70k
prediction_35k_ne = theta(1) + theta(2)*35000;
prediction_70k_ne = theta(1) + theta(2)*70000;

fprintf("\n\nThe prediction for profits in the city with 35000: %s",num2str(prediction_35k_ne));
fprintf("\nThe prediction for profits in the city with 70000: %s",num2str(prediction_70k_ne));

% ps2-4-h
% .0001,.001,.1,1 with 250 iters
[theta_0001, cost_0001] = gradientDescent(X, y, .0001, 250);
[theta_001, cost_001] = gradientDescent(X, y, .001, 250);
[theta_01, cost_01] = gradientDescent(X, y, .01, 250);
[theta_1, cost_1] = gradientDescent(X, y, .1, 250);
[theta1, cost1] = gradientDescent(X, y, 1.0, 250);

% Plot the first 3 graphs
figure(3);
plot(cost_0001);
hold on;
plot(cost_001);
plot(cost_01);
legend('alpha = .0001','alpha =  .001','alpha =  .01');
title("Cost vector over each iteration, alpha = .01, .001, .0001");
xlabel("Iteration");
ylabel("Cost");
hold off;

%Plot the last two on their own graphs
figure(4);
plot(cost_1);
title("Cost vector over each iteration, alpha = .1");
xlabel("Iteration");
ylabel("Cost");

figure(5);
plot(cost1);
title("Cost vector over each iteration, alpha = 1.0");
xlabel("Iteration");
ylabel("Cost");

% close all;

% ps2-5


% Separate the file by spaces into a matrix
fileID2 = fopen('input/hw2_data2.txt');
rawData2 = textscan(fileID,'%f %f %f','Delimiter',',');
fclose(fileID2);

% Set up the data 
X_raw1 = cell2mat(rawData2(:,1));
X_raw2 = cell2mat(rawData2(:,2));
X2 = ones(length(X_raw1), 2);
X2(:,2) = X_raw1;
X2(:,3) = X_raw2;
y2 = cell2mat(rawData2(:,3));

mean1 = mean(X2(:,2));
std1 = std(X2(:,2));
mean2 = mean(X2(:,3));
std2 = std(X2(:,3));

% normalize the data
fprintf("\n\nMean for feature 1: %s", num2str(mean(X2(:,2))));
fprintf("\nStandard deviation for feature 1: %s", num2str(std(X2(:,2))));

X2(:,2) = (X2(:,2) - mean(X2(:,2)) )/ std(X2(:,2));

fprintf("\n\nMean for feature 2: %s", num2str(mean(X2(:,3))));
fprintf("\nStandard deviation for feature 2: %s", num2str(std(X2(:,3))));

X2(:,3) = (X2(:,3) - mean(X2(:,3)) )/ std(X2(:,3));

% I was not sure if we were supposed to normalize the Y vector as well
% However, it made the theta vector rediculous numbers so I assumed not!
% y2(:) = (y2(:) - mean(y2(:)) )/ std(y2(:));


% Problem 5-a
% Use a learning rate of .01 and 1500 iterations and compute theta
[theta_part5a, cost_part5a] = gradientDescent(X2, y2, 0.01, 1500);

figure(6);
plot(cost_part5a);
title("Cost vector over each iteration with multiple variables");
xlabel("Iteration");
ylabel("Cost");

fprintf("\nTheta0 is %s, Theta1 is %s and Theta2 is %s",num2str(theta_part5a(1)),num2str(theta_part5a(2)),num2str(theta_part5a(3)));


% part c
% Predict price of house w/ 1650 sq ft and 3 bedrooms
% column 1 is size and column 2 is # of rooms

% normalize these values!
normalized_prediction_size = (1650 - mean1)/std1;
normalized_prediction_room = (3 - mean2)/std2;

price_prediction = theta_part5a(1) + theta_part5a(2)*normalized_prediction_size + theta_part5a(3)*normalized_prediction_room;

fprintf("\n\nThe prediction for a house with 1650 sqft and 3 bedrooms is %s", num2str(price_prediction));




