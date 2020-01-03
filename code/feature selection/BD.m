%特征选择部分：用二项分布进行特征选择
%特征提取频次法
%先验概率
%得到lncRNA矩阵之后
load('data.mat')%data中为每个4联体序列在每个样本中出现的次数
r1=size(data,1)
r2=size(data,2)

m1=sum(sum(data(1:280,:)));%m1为正样本中出现的所有4联体序列的数目之和
m2=sum(sum(data(281:840,:)));%m2为负样本中出现的所有4联体序列的数目之和
M=sum(sum(data));%m为所有样本中出现的所有4联体序列的数目之和
q1=m1/M;%q1表示在正样本中出现的所有序列的数目
q2=m2/M;%q2表示在负样本中出现的所有序列的数目
Q = [q1 q2];
%求nij（代表第i种特征在第j类样本中出现的概率）放到矩阵W里面,nij=W(i,j)

W=[];%生成一个256行*2列的数据集，每一行代表一个特征
for i=1:r2
    ni1 = sum(data(1:280,i));%第i种特征在正样本中的出现数目
    ni2 = sum(data(281:840,i));%第i种特征在负样本中的出现数目
    X = [ni1 ni2];
    W = [W;X];
end

%求Ni（所有样本中出现的第i种特征的数目）放到矩阵G
G = [];
for i = 1:r2
    j=sum(data(:,i));
    H = j;
    G = [G H];
end


%二项分布累加式，求盘p（nij）
DD=[];
for i=1:r2
   FF=[];
   for j=1:2
     sum = 0;
     for m=W(i,j):G(i)
          sum = binopdf(m,G(i),Q(j))+sum;
     end
     EE = sum;
     FF = [FF EE];%所有的p（nij）
     %FF=[p(ni1) p(ni2) p(ni3) p(ni4)]
    end
    DD = [DD;FF];
end

%DD是概率矩阵，每一行是第i个特征的在各类别中分别的概率，DD的维度是65536*4
%置信水平CL(ij)=1-p(nij)
CL = 1-DD;  
%CLi = max(CLi1,CLi2,CLi3,CLi4),找出每个特征在四类中最大的置信水平作为它的置信水平
[max_CL,index]=max(CL,[],2);
CLi = [max_CL,index];
CLimax = CLi(:,1);     %只剩下置信水平，不要index
Feorder = (1:r2)';  %列向量，从1到256
CLimax_order = [CLimax,Feorder];
%再从最大的置信水平开始倒序排列
CLimax_order = sortrows(CLimax_order,-1);   %降序排列
A2=CLimax_order(:,2)'