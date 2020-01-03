function [selectedArray] = featureSelect(data,A1,arraySize,a)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
F=[];
for j=1:arraySize
    E=data(:,A1(j));  
    F=[F E];
end
sortdata=F;
F1=[];
for j=1:a
    E=sortdata(:,j);
    F1=[F1 E];
end
    selectedArray=F1;%data4是经过特征在选择之后留下的个样本的特征集

end
    
