
myexercise=newindenpendentB;
myexercise=newindependent

    r1=size(myexercise,1);
    c=size(myexercise,2);
    %生成ACGT全排列矩阵
    mysequence=allSequence(5);%mysequence是生成的AGCT的4联体核苷酸的所有排列组合
    mysequence=char(mysequence);
    r2=size(mysequence,1);
    data=[];
    for i=1:r1%r1是样本集的行数
        F=[];
        for k=1:r2%r2是AGTC所有排列组合的行数
            m=0;
            for j=1:c-5+1%c是样本集的列数
                 if [myexercise(i,j) myexercise(i,j+1) myexercise(i,j+2) myexercise(i,j+3) myexercise(i,j+4)]==[mysequence(k,1) mysequence(k,2) mysequence(k,3) mysequence(k,4) mysequence(k,5)]
                  m=m+1;        
                 end
            end
            E=m;
            F=[F E];
        end
        data=[data;F];
    end 
    sum=ones(r1,1)
for i=1:r1
    sum(i,1)=length(deblank(myexercise(i,:)))-5+1
    data(i,:)=data(i,:)/sum(i,1)
end
data1=data
%3、提取特征
    %整理出要编码的数据集  
load('mysequencenum.mat')
load('phy.mat')
k=5
z=4^k
d=zeros(r1,30);
for n=1:r1
    len=size(deblank(myexercise(n,:)),2); 
    for a=1:5%r1是样本集的行数  
        for e=1:6%r2是AGTC所有排列组合的行数
            t=0
            for i=1:(len-k-a)
               t=t+J(myexercise,n,e,i,a,mysequence,pyh);
            end
            t=t/(len-k-a);
           
            d(n,e+(a-1)*6)=t;
           
        end
    end
end

% selectedfeature=featureSelect(data,A,1024,883)
data=[data d]