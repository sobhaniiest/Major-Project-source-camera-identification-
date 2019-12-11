function findSPN()
clc;
clear all;
addpath('.\Filter');
addpath('.\Functions');
RootDirName = 'F:\Major Project\Vision Dataset';
dlist = dir(RootDirName);
OutputDirName = 'F:\Major Project\test_wa\SPN_WA';

for i=19:37 %length(dlist)
    dirName = strcat(RootDirName,'\',dlist(i).name,'\images');
    OutputdName = strcat(OutputDirName,'\',dlist(i).name);
    mkdir(OutputdName);
    %ModelName = {'flat', 'nat', 'natFBH', 'natFBL', 'natWA'};
    ModelName = {'natWA'};
    fprintf('Extracting fingerPrint (SPN) for the camera model "%s"\n',dlist(i).name);
    
    for k=1:length(ModelName)
        dname = strcat(dirName,'\',ModelName{k},'\*.jpg');
        tList = dir(dname);
        
        %fprintf('Extracting fingerPrint (SPN) for the camera model "%s"\n',dname);  
        fprintf('Extracting fingerPrint (SPN) for the "%s" images\n',ModelName{k});
        
        ImageListPath = {};
        RefPRNU = [];
        for j=1:100 %length(tList)       
            ImageListPath = [ImageListPath; strcat(dirName,'\',...
                ModelName{k},'\',tList(j).name)];
        end    
        RefPRNU = single(getFingerprint({ImageListPath{:}}));    
        filename1=strcat(OutputdName,'\',ModelName{k},'.mat');
        save(filename1, 'RefPRNU');
        
        clear ImageListPath
        clear RefPRNU
        clear filename1
        
    end
end
