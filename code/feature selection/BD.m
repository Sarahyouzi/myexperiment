%����ѡ�񲿷֣��ö���ֲ���������ѡ��
%������ȡƵ�η�
%�������
%�õ�lncRNA����֮��
load('data.mat')%data��Ϊÿ��4����������ÿ�������г��ֵĴ���
r1=size(data,1)
r2=size(data,2)

m1=sum(sum(data(1:280,:)));%m1Ϊ�������г��ֵ�����4�������е���Ŀ֮��
m2=sum(sum(data(281:840,:)));%m2Ϊ�������г��ֵ�����4�������е���Ŀ֮��
M=sum(sum(data));%mΪ���������г��ֵ�����4�������е���Ŀ֮��
q1=m1/M;%q1��ʾ���������г��ֵ��������е���Ŀ
q2=m2/M;%q2��ʾ�ڸ������г��ֵ��������е���Ŀ
Q = [q1 q2];
%��nij�������i�������ڵ�j�������г��ֵĸ��ʣ��ŵ�����W����,nij=W(i,j)

W=[];%����һ��256��*2�е����ݼ���ÿһ�д���һ������
for i=1:r2
    ni1 = sum(data(1:280,i));%��i���������������еĳ�����Ŀ
    ni2 = sum(data(281:840,i));%��i�������ڸ������еĳ�����Ŀ
    X = [ni1 ni2];
    W = [W;X];
end

%��Ni�����������г��ֵĵ�i����������Ŀ���ŵ�����G
G = [];
for i = 1:r2
    j=sum(data(:,i));
    H = j;
    G = [G H];
end


%����ֲ��ۼ�ʽ������p��nij��
DD=[];
for i=1:r2
   FF=[];
   for j=1:2
     sum = 0;
     for m=W(i,j):G(i)
          sum = binopdf(m,G(i),Q(j))+sum;
     end
     EE = sum;
     FF = [FF EE];%���е�p��nij��
     %FF=[p(ni1) p(ni2) p(ni3) p(ni4)]
    end
    DD = [DD;FF];
end

%DD�Ǹ��ʾ���ÿһ���ǵ�i���������ڸ�����зֱ�ĸ��ʣ�DD��ά����65536*4
%����ˮƽCL(ij)=1-p(nij)
CL = 1-DD;  
%CLi = max(CLi1,CLi2,CLi3,CLi4),�ҳ�ÿ����������������������ˮƽ��Ϊ��������ˮƽ
[max_CL,index]=max(CL,[],2);
CLi = [max_CL,index];
CLimax = CLi(:,1);     %ֻʣ������ˮƽ����Ҫindex
Feorder = (1:r2)';  %����������1��256
CLimax_order = [CLimax,Feorder];
%�ٴ���������ˮƽ��ʼ��������
CLimax_order = sortrows(CLimax_order,-1);   %��������
A2=CLimax_order(:,2)'