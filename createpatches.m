
clear all
close all
clc
inputfolder='./Test/Input/';
 %dataname=dir('./JPEGImages/*jpg');

 
for i=2
      ccnt=1;
       a=imresize(imread([inputfolder num2str(i) '.jpg' ]),[768,768],'bicubic');

%        mkdir(['./Test_patches/' num2str(i)]);
        [blocksa,idx] = im2blocks(a,[128,128],32);
%           for kl=1:length(idx)
%              Apatch=reshape(blocksa(:,kl),[128,128]);
%               
%                  imwrite(uint8(Apatch),['./Test_patches/' num2str(i) '/' num2str(ccnt)  '.jpg'])
%                   ccnt=ccnt+1;
%           end
% end

      for kl=1:length(idx)
       blocksd(:,kl)=reshape(double(rgb2gray(imread(['/Results/40000_net_G_' num2str(i) '/images/output/' num2str(kl) '.jpg']))),[128*128,1]);
      end

    bb=[128,128];
    count = 1;
Weight = zeros(size(a,1),size(a,2));
IMout = zeros(size(a,1),size(a,2));
IMout1 = zeros(size(a,1),size(a,2));
[rows,cols] = ind2sub(size(a)-bb+1,idx);
for ii= 1:length(cols)
    col = cols(ii); row = rows(ii);
    block1 =reshape(blocksd(:,count),bb);
    IMout(row:row+bb(1)-1,col:col+bb(2)-1,:)=IMout(row:row+bb(1)-1,col:col+bb(2)-1,:)+block1;
    Weight(row:row+bb(1)-1,col:col+bb(2)-1)=Weight(row:row+bb(1)-1,col:col+bb(2)-1)+ones(bb(1));
    count = count+1;
end

IOut = (IMout)./repmat((Weight),[1,1,3]);
imwrite(uint8(imresize(a,[512,512],'bicubic')),['./Out/Inp' num2str(i) '.png']);
imwrite(uint8(IOut),['./Out/G2_' num2str(i) '.png'])
Outp=imread(['./Test/Target/' num2str(i) '.jpg']);
imwrite(Outp,['./Out/Tar' num2str(i) '.png']);
end
   