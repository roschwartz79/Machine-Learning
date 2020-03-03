function maxIndex = maxScore(vector2calc)
% What is the max index
maxIndex = 1;
maxValue = -100;
for i = 1:40
   if vector2calc(i) > 0 && vector2calc(i) > maxValue
       maxValue = vector2calc(i);
       maxIndex = i;
   end
end

end