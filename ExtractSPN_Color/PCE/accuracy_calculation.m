clc;
clear all;

DeviceDirName = 'F:\Major Project\Vision Dataset';
PCEDirName = 'F:\Major Project\PCE_natWA';
thre_file = fopen('threshold.txt', 'r');
THRE = fscanf(thre_file, '%f');

dlist = dir(DeviceDirName);
PCElist = dir(PCEDirName);

count = 0;
k = 0;
for i=3:17 %length(dlist)
    %if i ~= 18
        k = k + 1;
        AccuracyMatrix=zeros(100,length(PCElist) - 2);
        PCEName = strcat(PCEDirName,'\',PCElist(i).name);
        %disp(dlist(i).name)
        disp(PCElist(i).name)

        table = xlsread(PCEName);
        thr = THRE(i-2);

        for y=1:15%35
            %if y ~= 16
                for x=1:100 
                    if thr < table(x, y)
                        AccuracyMatrix(x, y) = 1;
                    end
                end
            %end
        end
        
        for x=1:100
            flag = 0;
            if AccuracyMatrix(x, i-2) == 1 
                for y=1:15 %35
                    if y ~= i-2 && AccuracyMatrix(x, y) == 1 %&& y ~= 16
                        flag = 1;
                    end
                end
                if flag == 0
                    count = count + 1;
                end
            end
        end
                    
        
        %filenam = strcat('F:\Major Project\accuracy\PCE_nat\', dlist(i).name, '.xlsx');
        filenam = strcat('F:\Major Project\test_wa\', dlist(i).name, '.xlsx');
        xlswrite(filenam, AccuracyMatrix);
        clear filenam;
        clear AccuracyMatrix;
        clear table;
    %end
end

fprintf('\nAccuracy = %f\n', count/15);
disp(k);