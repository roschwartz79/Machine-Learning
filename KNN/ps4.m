% ECE 1395 Homework 4
% Written by Rob Schwartz

% load the data
data1 = load("input/hw4_data1.mat");
data2 = load("input/hw4_data2.mat");

% Part 1- effect of K

% Set up the training data
x_train1 = [data1.X1 ; data1.X2 ; data1.X3 ; data1.X4];
x_train2 = [data1.X1 ; data1.X2 ; data1.X3 ; data1.X5];
x_train3 = [data1.X1 ; data1.X2 ; data1.X4 ; data1.X5];
x_train4 = [data1.X1 ; data1.X3 ; data1.X4 ; data1.X5];
x_train5 = [data1.X2 ; data1.X3 ; data1.X4 ; data1.X5];

x_train_vector = [x_train1 , x_train2 , x_train3 , x_train4 , x_train5];

y_train1 = [data1.y1 ; data1.y2 ; data1.y3 ; data1.y4];
y_train2 = [data1.y1 ; data1.y2 ; data1.y3 ; data1.y5];
y_train3 = [data1.y1 ; data1.y2 ; data1.y4 ; data1.y5];
y_train4 = [data1.y1 ; data1.y3 ; data1.y4 ; data1.y5];
y_train5 = [data1.y2 ; data1.y3 ; data1.y4 ; data1.y5];

y_train_vector = [y_train1 , y_train2 , y_train3 , y_train4 , y_train5];

% Store the data in a vector-
% 8 rows for the 8 values of k
% each column contains 1 of the folds (5 folds)
trained_model_vector = {8,5};
label_vector = {8,5};
accuracy_vector = {8,5};
average_accuracy = zeros(8,1);

%loop through each value of k
index = 1;
for k = 1:2:15
    % each fold
    
    trained_model_vector(index,1) = {fitcknn(x_train1,y_train1,'NumNeighbors',k)};
    trained_model_vector(index,2) = {fitcknn(x_train2,y_train2,'NumNeighbors',k)};
    trained_model_vector(index,3) = {fitcknn(x_train3,y_train3,'NumNeighbors',k)};
    trained_model_vector(index,4) = {fitcknn(x_train4,y_train4,'NumNeighbors',k)};
    trained_model_vector(index,5) = {fitcknn(x_train5,y_train5,'NumNeighbors',k)};
    
    index = index + 1;
end

% get the predictions and accuracy
for i = 1:8
    label_vector(i,1) = {predict(trained_model_vector{i,1},data1.X5)}; 
    label_vector(i,2) = {predict(trained_model_vector{i,2},data1.X4)}; 
    label_vector(i,3) = {predict(trained_model_vector{i,3},data1.X3)}; 
    label_vector(i,4) = {predict(trained_model_vector{i,4},data1.X2)}; 
    label_vector(i,5) = {predict(trained_model_vector{i,5},data1.X1)}; 
    
    accuracy_vector(i,1) = {getAccuracy(label_vector{i,1},data1.y5)};
    accuracy_vector(i,2) = {getAccuracy(label_vector{i,2},data1.y4)};
    accuracy_vector(i,3) = {getAccuracy(label_vector{i,3},data1.y3)};
    accuracy_vector(i,4) = {getAccuracy(label_vector{i,4},data1.y2)};
    accuracy_vector(i,5) = {getAccuracy(label_vector{i,5},data1.y1)};
    
    average_accuracy(i) = (accuracy_vector{i,1} + accuracy_vector{i,2} + accuracy_vector{i,3} + accuracy_vector{i,4} + accuracy_vector{i,5})/5;
end


% Plot the data
figure(1);
plot(average_accuracy);
title("Average accuracy vs K");
xlabel("Value of K");
xticks([1,2,3,4,5,6,7,8]);
xticklabels({'1','3','5','7','9','11','13','15'});
ylabel("Average accuracy");


% Question 2

% setup the data
X_train = data2.X_train;
y_train = data2.y_train;
X_test = data2.X_test;
y_test = data2.y_test;

% Test the data
label_y_01 = weightedKNN(X_train, y_train, X_test, .1);
label_y_05 = weightedKNN(X_train, y_train, X_test, .5);
label_y_1 = weightedKNN(X_train, y_train, X_test, 1);
label_y_3 = weightedKNN(X_train, y_train, X_test, 3);
label_y_5 = weightedKNN(X_train, y_train, X_test, 5);

getAccuracy(label_y_01,y_test)
getAccuracy(label_y_05,y_test)
getAccuracy(label_y_1,y_test)
getAccuracy(label_y_3,y_test)
getAccuracy(label_y_5,y_test)

fprintf("\nAccuracy vs Sigma");
fprintf("\n\n   .1          .5            1          3           5");
fprintf("\n%f    %f    %f    %f    %f",getAccuracy(label_y_01,y_test),getAccuracy(label_y_05,y_test),getAccuracy(label_y_1,y_test),getAccuracy(label_y_3,y_test),getAccuracy(label_y_5,y_test));















