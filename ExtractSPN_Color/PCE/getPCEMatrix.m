clc;
clear all;
close all;
addpath('.\Filter');
addpath('.\Functions');

RootDirName = 'F:\Major Project\Vision Dataset';
dlist = dir(RootDirName);

%total_imgPerModel = 100;
%finalPCEMatrix=zeros(100*length(dlist),length(dlist));

for j=16:16  %length(dlist)
    finalPCEMatrix=zeros(100,length(dlist) - 2);
    dirName = strcat(RootDirName,'\',dlist(j).name,'\images\natWA','\*.jpg');
    list_img = dir(dirName);
    fprintf('Camera model "%s"\n',dlist(j).name);
              
    for im=1:100
        filename = strcat(RootDirName,'\',dlist(j).name,'\images\natWA','\',list_img(im).name);
        imx = imread(filename);
        imx=imcrop(imx,[1,1,511,511]);
        disp(im)
              
        for i=3:length(dlist)
            if i ~= 18
                ModelSPN=strcat('F:\Major Project\test_wa\SPN_WA\',dlist(i).name,'\natWA.mat');
                RP1=load(ModelSPN);
                RP=RP1.RefPRNU;
                RP=RP(1:512,1:512);
                RP = rgb2gray1(RP);
                sigmaRP = std2(RP);
                Fingerprint = WienerInDFT(RP,sigmaRP);
                Noisex = NoiseExtractFromImage(imx,2);
                Noisex = WienerInDFT(Noisex,std2(Noisex));
                Ix=double(rgb2gray(imx));
                C = crosscorr(Noisex,Ix.*Fingerprint);
                detection = PCE(C);
                %finalPCEMatrix(im+(j-1)*total_imgPerModel,i)=detection.PCE;
                finalPCEMatrix(im,i-2)=detection.PCE;
            end
        end
    end
    
    filenam = strcat('F:\Major Project\test_wa\PCE_natWA\', dlist(j).name, '.xlsx');
    xlswrite(filenam, finalPCEMatrix);
    clear filenam;
    clear finalPCEMatrix;
end


