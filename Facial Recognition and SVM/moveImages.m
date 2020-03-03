function moveImages
% Delete all images from last run
delete('input/train/*.*');
delete('input/test/*.*');

% loop through 40 people
for i = 1:40
    
    % data structure to track what has been used so far
    % 1 means not taken, 0 means taken
    tracker = ones(1,10);
    % loop through 10 images per person
    for j = 1:8
       % get an image we haven't taken yet
       pic2take = randi(10);
       while tracker(pic2take) == 0
           pic2take = randi(10);
       end
       tracker(pic2take) = 0;
       copystring = "input/all/s" + i + "/" + pic2take + ".pgm";
       pastestring = "input/train/s" + i + j + ".pgm";
       copyfile(copystring, pastestring);
    end
    % Now put the testing pics in testing folder
    mkdir("input/test/s" + i);
    t = 1;
    for j = 1:10
       if tracker(j) == 1
           copystring = "input/all/s" + i + "/" + j + ".pgm";
           pastestring = "input/test/s" + i + "/image" + t + ".pgm";
           t = t + 1;
           copyfile(copystring, pastestring);
       end
    end
end



end