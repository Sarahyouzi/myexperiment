%1、导入数据
M=csvread('D:\\data\\cor.csv')
cor=cell(11943,11943)
    terminators=cell(560,1)
    nonterminators=cell(1120,1)
    terminatorE=cell(294,1)
    terminatorB=cell(850,1)
    cor=textread('D:\\data\\cor.csv','%s')
    terminators=textread('2018-bioinformatics-a sequence-based tool-dataset(2)280 terminators.csv','%s')
    nonterminators=textread('2018-bioinformatics-a sequence-based tool-dataset560non- terminators.csv','%s')
    terminatorE=textread('2018-bioinformatics-a sequence-based tool-dataset(1)147rho-independent terminators in E. coli.csv','%s')
    terminatorB=textread('2018-bioinformatics-a sequence-based tool-dataset(3)425 rho-independent terminators in B. subtilis.csv','%s')
%2、整理数据
for i=1:1:length(terminators)/2
    terminators(i,:)=[];
end
for i=1:1:length(nonterminators)/2
   nonterminators(i,:)=[];
end
 
    %myexercise是正负样本的集合，用来当做测试集
    myexercise(1:280,:)=terminators;
    myexercise(281:840,:)=nonterminators;
    myexercise=char(myexercise);
    
 