%% S1-156
%1、导入数据
   
    terminators=cell(560,1)
    nonterminators=cell(1120,1)
    terminatorE=cell(294,1)
    terminatorB=cell(850,1)
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
for i=1:1:length(terminatorE)/2
    terminatorE(i,:)=[];
end
for i=1:1:length(terminatorB)/2
   terminatorB(i,:)=[];
end
    myexercise=cell(840,1);%myexercise是正负样本的集合，用来当做测试集
    myexercise(1:280,:)=terminators;
    myexercise(281:840,:)=nonterminators;
    myexercise=char(myexercise);
    terminatorB=char(terminatorB)
    terminatorE=char(terminatorE)
    r1=size(myexercise,1);



A=zeros(r1,1)%A的含量
C=zeros(r1,1)%C的含量
G=zeros(r1,1)%G的含量
T=zeros(r1,1)%T的含量
len=zeros(r1,1)%每个序列的长度
AT=zeros(r1,1)% 1 A+T的含量
A_T=zeros(r1,1)%A-T的含量
GC=zeros(r1,1)% 2 G+C的含量
G_C=zeros(r1,1)%G-C的含量
GdC=zeros(r1,1)% 3 G/C
gc=zeros(r1,1)% 4 G-C/G+C
at=zeros(r1,1)% 5 A-T/A+T
ac=zeros(r1,1)% 6 A+T/G+C
for i=1:r1%r1是样本集的行数
    len(i,1)=size(deblank(myexercise(i,:)),2);
    for j=1:len(i,1)%r2是AGTC所有排列组合的行数
        if myexercise(i,j)=='C'
            C(i,1)=C(i,1)+1;        
        end
        if myexercise(i,j)=='G'
            G(i,1)=G(i,1)+1;        
        end
        if myexercise(i,j)=='A'
            A(i,1)=A(i,1)+1;        
        end
         if myexercise(i,j)=='T'
            T(i,1)=T(i,1)+1;        
        end
        
    end
        GC(i,1)=(G(i,1)+C(i,1))/len(i,1);
        AT(i,1)=(A(i,1)+T(i,1))/len(i,1);
        A_T(i,1)=A(i,1)-T(i,1);
        G_C(i,1)=G(i,1)-C(i,1);
        gc(i,1)=G_C(i,1)/GC(i,1)
        at(i,1)=A_T(i,1)/AT(i,1)
        ac(i,1)=AT(i,1)/GC(i,1)
        GdC(i,1)=C(i,1)/G(i,1);
end
data=[AT GC at gc ac GdC]

