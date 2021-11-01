function [blocks,idx] = im2blocks(I1,bb,shift)

if (shift==1)
    blocks = im2col(I1,[bb bb],'sliding');
    idx = [1:size(blocks,2)];
else
 idxMat = zeros(size(I1)-bb+1);
idxMat([1:shift:end-1,end],[1:shift:end-1,end]) = 1; % take blocks in distances of 'slidingDix', but always take the first and last one (in each row and column).
idx = find(idxMat);
[rows,cols] = ind2sub(size(idxMat),idx);
blocks = zeros(prod(bb),length(idx));
for i = 1:length(idx)
    currBlock = I1(rows(i):rows(i)+bb(1)-1,cols(i):cols(i)+bb(2)-1,:);
    blocks(:,i) = currBlock(:);
end
end