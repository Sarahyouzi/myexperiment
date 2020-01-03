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

%3、提取特征
sum=ones(r1,1)
for i=1:r1
    sum(i,1)=length(deblank(myexercise(i,:)))-5+1
    data(i,:)=data(i,:)/sum(i,1)
end
data1=data
    %整理出要编码的数据集  
    myexercise=newindenpendentB;
    myexercise=newindependent
    r1=size(myexercise,1);
    c=size(myexercise,2);
    mysequence=allSequence(5);%mysequence是生成的AGCT的4联体核苷酸的所有排列组合
    mysequence=char(mysequence);
     r2=size(mysequence,1);
load('mysequencenum.mat')
load('phy.mat')
k=5
z=4^k

%前半部分
for n=1:r1
    len=size(deblank(myexercise(n,:)),2); 
    sita=0
    for i=1:5
        sum=0
        for j=1:(len-1-i)
            sum1=0
            for p=1:6   
               sum1=sum1+power((mysequencenum( myexercise(n,j:j+1),p,mysequence,pyh)-mysequencenum( myexercise(n,j+i:j+1+i),p,mysequence,pyh)),2) 
            end
            sum1=sum1/6
            sum=sum+sum1
        end
        sum=sum/(len-1-i)
      sita=sita+sum
    end
    data(n,:)=data(n,:)/(1+sita)
end

d=zeros(r1,5);

for r=1:r1
     len=size(deblank(myexercise(r,:)),2); 
    sita=0
    for i=1:5
        sum=0
        for j=1:(len-1-i)
            sum1=0
            for p=1:6   
               sum1=sum1+power((mysequencenum( myexercise(r,j:j+1),p,mysequence,pyh)-mysequencenum( myexercise(r,j+i:j+1+i),p,mysequence,pyh)),2) 
            end
            sum1=sum1/6
            sum=sum+sum1
        end
        sum=sum/(len-1-i)
      sita=sita+sum
    end
    for i=1:5
        sum=0
        for j=1:(len-1-i)
            sum1=0
            for p=1:6   
               sum1=sum1+power((mysequencenum( myexercise(r,j:j+1),p,mysequence,pyh)-mysequencenum( myexercise(r,j+i:j+1+i),p,mysequence,pyh)),2) 
            end
            sum1=sum1/6
            sum=sum+sum1
        end
        sum=sum/(len-1-i)
        d(r,i)=sum/(sita+1)
    end
end
data=[data d]