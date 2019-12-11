clc;
clear all;

file = fopen('dataset.txt', 'w');

min_x = 100000;
min_y = 100000;

RootDirName = 'F:\Major Project\Vision Dataset';
dlist = dir(RootDirName);

for i=3:length(dlist)
    dirName = strcat(RootDirName,'\',dlist(i).name,'\images');
    disp(dlist(i).name)
    fprintf(file,'%s\n',dlist(i).name);
    ModelName = {'flat', 'nat', 'natFBH', 'natFBL', 'natWA'};

    for k=1:length(ModelName)
        disp(ModelName{k})
        
        dname = strcat(dirName,'\',ModelName{k},'\*.jpg');
        tList = dir(dname);
        disp(length(tList))
        fprintf(file,'%s - %d\n',ModelName{k}, length(tList));
        
        for j=1:length(tList)       
            image = strcat(dirName,'\',ModelName{k},'\',tList(j).name);
            im = imread(image);
            [x, y, z] = size(im);
            if x  < min_x
                min_x = x;
            end
            if x  < min_y
                min_y = y;
            end
        end   
    end 
end

disp(min_x)
disp(min_y)

fprintf(file,'min_x : %d - min_y : %d\n',min_x, min_y);

