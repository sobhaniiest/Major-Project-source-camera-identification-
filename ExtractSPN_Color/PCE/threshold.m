clc;
clear all;

file = fopen('threshold_nat.txt', 'w');

DeviceDirName = 'F:\Major Project\Vision Dataset';
PCEDirName = 'F:\Major Project\PCE_NAT';
thre_file = fopen('threshold.txt', 'w');

dlist = dir(DeviceDirName);
PCElist = dir(PCEDirName);

for i=3:length(dlist)
        PCEName = strcat(PCEDirName,'\',PCElist(i).name);
        disp(dlist(i).name)
        disp(PCElist(i).name)
        min = 1000000000;
        max = -1000000000;

        table = xlsread(PCEName);

        for j=1:100
            if min > table(j, i-2)
                min = table(j, i-2);
            end
        end

        for x=1:100
            for y=1:35
                if y ~= i-2
                    if max < table(x, y)
                        max = table(x, y);
                    end
                end
            end 
        end
        
        disp(min)
        disp(max)
        
        fprintf(file, '%s \n\nmin : %f \nmax : %f\n\n', dlist(i).name, min, max);

        if min > max
            thr = min;
        else
            thr = (min + max ) / 2;
        end

        disp(thr)
        fprintf(file,'Threshold : %f\n\n', thr);
        fprintf(thre_file,'%f\n', thr);

        clear table
end


