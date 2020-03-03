function accuracy = getAccuracy(prediction, actual)
misclassified = 0;
accuracy = 0;
for i = 1:size(prediction)
    if prediction(i) ~= actual(i)
        misclassified = misclassified + 1;
    end
end

accuracy = (size(prediction)-misclassified)/size(prediction);

end