clc;
clear all;
close all;
addpath('.\Filter');
addpath('.\Functions');
% ModelName = {'CanonA640','Kodak','NikonD70','Olympus','Panasonic','Praktica','Ricoh','Rollei','Samsung74','SonyH50'};
ModelName = {'Rollei','Samsung74','SonyH50'};

RootDirName = 'E:\Counter Forensic Images\Inpaint_Images';

% dirName = strcat(RootDirName,'\',ModelName{1},'\*.jpg');
% list_img = dir(dirName);
 total_imgPerModel = 100;
finalPCEMatrix=zeros(100*length(ModelName),length(ModelName));

for j=1:length(ModelName)
              dirName = strcat(RootDirName,'\',ModelName{j},'\*.jpg');
              inDir=strcat(RootDirName,'\',ModelName{j});
              list_img = dir(dirName);
              
            for im=1:100
              filename = strcat(inDir,'\',list_img(im).name);
              imx = imread(filename);
              
              imx=imcrop(imx,[1,1,511,511]);
              for i=1:length(ModelName)
                  ModelSPN=strcat('./SPN/',ModelName{i},'.mat');
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
                  finalPCEMatrix(im+(j-1)*total_imgPerModel,i)=detection.PCE;
              end
            end
    
xlswrite('PCE-Results.xlsx',finalPCEMatrix);
end
