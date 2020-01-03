% load('C:\Users\Lv\Desktop\data\Human Activity\train_label.txt')
% save('F:\MATLAB\R2015b\bin\Adaboost\train_label1.mat','train_label')

% .fscore 得分矩阵的一部分
A=textread('D:\代码文件\实验文件\5联体核苷酸\data1_tzj31.fscore.txt','%s','delimiter',':');
A=textread('D:\代码文件\实验文件\5联体核苷酸\data1_pseknc2.5.fscore.txt','%s','delimiter',':');
A=textread('D:\代码文件\实验文件\5联体核苷酸\data1_property50.fscore.txt','%s','delimiter',':');
a=size(A,1)/2-1;     %带标签的特征训练集的行数除以2再减1，也就是数据集的一半的数目a
C=[];
for i=0:a
    D11=A(2*i+1,:);  %读奇数行
    C=[C D11];       %水平并排
end

A=str2double(C);    %取数据集的一半，再取一半数据集的奇数行，str类型转成double类型
add=ones(1,3)

% for i=1:3
%     add(1,i)=1029+i 
% end
% A34=[add A3]

%选取特定的维度，存入C中
% a=size(A,1)/2-1;
% C=[];
% for i=0:a
%     D11=A(2*i+1,:);
%     C=[C D11]; 
% end
% C1=char(C)
% D=double(1:256)
% % A1=str2double(C);
% %整理数据，并将其变为数值
% for i=1:size(C1,1)
%     str=''
%     for j=1:4
%         if C1(i,j)==':'
%             break
%         end
%         str=strcat(str,C1(i,j))
%     end
%     D(i)=str2double(str)
% end
% sortdata=D