%% S1-156
%1����������
    terminators=cell(560,1)
    nonterminators=cell(1120,1)
    terminatorE=cell(294,1)
    terminatorB=cell(850,1)
    terminators=textread('2018-bioinformatics-a sequence-based tool-dataset(2)280 terminators.csv','%s')
    nonterminators=textread('2018-bioinformatics-a sequence-based tool-dataset560non- terminators.csv','%s')
    terminatorE=textread('2018-bioinformatics-a sequence-based tool-dataset(1)147rho-independent terminators in E. coli.csv','%s')
    terminatorB=textread('2018-bioinformatics-a sequence-based tool-dataset(3)425 rho-independent terminators in B. subtilis.csv','%s')
%2����������
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
    myexercise=cell(840,1);%myexercise�����������ļ��ϣ������������Լ�
    myexercise(1:280,:)=terminators;
    myexercise(281:840,:)=nonterminators;
    myexercise=char(myexercise);
    terminatorB=char(terminatorB)
    terminatorE=char(terminatorE)
    myexercise=terminatorB
    r1=size(myexercise,1);
    c=size(myexercise,2);
F=zeros(4,172)%ACGT,��������ĳ��ִ���
aa=zeros(1,172)%ÿһ��λ�ó��ֵĺ����������Ŀ
for i=1:r1%r1��������������
        for j=1:c%�������е���
                 if myexercise(i,j)=='A'
                      F(1,j)=F(1,j)+1;
                      aa(1,j)=aa(1,j)+1
                 end
                 if myexercise(i,j)=='C'
                      F(2,j)=F(2,j)+1;
                      aa(1,j)=aa(1,j)+1
                 end
                 if myexercise(i,j)=='G'
                      F(3,j)=F(3,j)+1;
                      aa(1,j)=aa(1,j)+1
                 end
                 if myexercise(i,j)=='T'
                      F(4,j)=F(4,j)+1;
                      aa(1,j)=aa(1,j)+1
                 end      
        end
end
FF=zeros(4,172)%ACGT���ֵ�Ƶ��
for i=1:4
    for j=1:172
        FF(i,j)=F(i,j)/aa(1,j)
    end
end
% P=ones(840,1)%ÿһ�����еĸ���
% 
% for i=1:r1%r1��������������
%         for j=1:c%�������е���
%                  if myexercise(i,j)=='A'
%                      P(i,1)=P(i,1)*FF(1,j)
%                  end
%                  if myexercise(i,j)=='C'
%                       P(i,1)=P(i,1)*FF(2,j)
%                  end
%                  if myexercise(i,j)=='G'
%                       P(i,1)=P(i,1)*FF(3,j)
%                  end
%                  if myexercise(i,j)=='T'
%                       P(i,1)=P(i,1)*FF(4,j)
%                  end      
%         end
% end

PWM1=ones(4,172)%λ��Ȩ�ؾ���
for i=1:4
    for j=1:172
        PWM1(i,j)=log2(FF(i,j)/0.25)
    end
end
%Ϊ������г��Ȳ�һ�µ���������ȡ��������
F1=zeros(4,3)
aa1=zeros(1,3)
for i=1:r1%r1��������������
        for j=1:39%�������е���
                 if myexercise(i,j)=='A'
                      F1(1,1)=F1(1,1)+1;
                      aa1(1,1)=aa1(1,1)+1
                 end
                 if myexercise(i,j)=='C'
                      F1(2,1)=F1(2,1)+1;
                      aa1(1,1)=aa1(1,1)+1
                 end
                 if myexercise(i,j)=='G'
                      F1(3,1)=F1(3,1)+1;
                      aa1(1,1)=aa1(1,1)+1
                 end
                 if myexercise(i,j)=='T'
                      F1(4,1)=F1(4,1)+1;
                      aa1(1,1)=aa1(1,1)+1
                 end      
        end
        for j=40:80%�������е���
                 if myexercise(i,j)=='A'
                      F1(1,2)=F1(1,2)+1;
                      aa1(1,2)=aa1(1,2)+1
                 end
                 if myexercise(i,j)=='C'
                      F1(2,2)=F1(2,2)+1;
                      aa1(1,2)=aa1(1,2)+1
                 end
                 if myexercise(i,j)=='G'
                      F1(3,2)=F1(3,2)+1;
                      aa1(1,2)=aa1(1,2)+1
                 end
                 if myexercise(i,j)=='T'
                      F1(4,2)=F1(4,2)+1;
                      aa1(1,2)=aa1(1,2)+1
                 end      
        end
        for j=81:172%�������е���
                 if myexercise(i,j)=='A'
                      F1(1,3)=F1(1,3)+1;
                      aa1(1,3)=aa1(1,3)+1
                 end
                 if myexercise(i,j)=='C'
                      F1(2,3)=F1(2,3)+1;
                      aa1(1,3)=aa1(1,3)+1
                 end
                 if myexercise(i,j)=='G'
                      F1(3,3)=F1(3,3)+1;
                      aa1(1,3)=aa1(1,3)+1
                 end
                 if myexercise(i,j)=='T'
                      F1(4,3)=F1(4,3)+1;
                      aa1(1,3)=aa1(1,3)+1
                 end      
        end
end
FF1=zeros(4,3)%ACGT���ֵ�Ƶ��
for i=1:4
    for j=1:3
        FF1(i,j)=F1(i,j)/aa1(1,j)
    end
end
PWM2=ones(4,3)%λ��Ȩ�ؾ���
for i=1:4
    for j=1:3
        PWM2(i,j)=log2(FF1(i,j)/0.25)
    end
end

add=zeros(840,1)%ÿһ�����еĸ���
for i=1:r1%r1������������
        for j=1:c%�������е���
                 if myexercise(i,j)=='A'
                     add(i,1)=add(i,1)+PWM1(1,j)
                 end
                 if myexercise(i,j)=='C'
                      add(i,1)=add(i,1)+PWM1(2,j)
                 end
                 if myexercise(i,j)=='G'
                      add(i,1)=add(i,1)+PWM1(3,j)
                 end
                 if myexercise(i,j)=='T'
                      add(i,1)=add(i,1)+PWM1(4,j)
                 end      
        end
       
end
% for i=1:r1%r1��������������
%         for j=1:39%�������е���
%                  if myexercise(i,j)=='A'
%                      add(i,1)=add(i,1)+PWM1(1,j)+PWM2(1,1)
%                  end
%                  if myexercise(i,j)=='C'
%                       add(i,1)=add(i,1)+PWM1(2,j)+PWM2(2,1)
%                  end
%                  if myexercise(i,j)=='G'
%                       add(i,1)=add(i,1)+PWM1(3,j)+PWM2(3,1)
%                  end
%                  if myexercise(i,j)=='T'
%                       add(i,1)=add(i,1)+PWM1(4,j)+PWM2(4,1)
%                  end      
%         end
%         for j=40:80%�������е���
%                  if myexercise(i,j)=='A'
%                      add(i,1)=add(i,1)+PWM1(1,j)+PWM2(1,2)
%                  end
%                  if myexercise(i,j)=='C'
%                       add(i,1)=add(i,1)+PWM1(2,j)+PWM2(2,2)
%                  end
%                  if myexercise(i,j)=='G'
%                       add(i,1)=add(i,1)+PWM1(3,j)+PWM2(3,2)
%                  end
%                  if myexercise(i,j)=='T'
%                       add(i,1)=add(i,1)+PWM1(4,j)+PWM2(4,2)
%                  end      
%         end
%         for j=80:172%�������е���
%                  if myexercise(i,j)=='A'
%                      add(i,1)=add(i,1)+PWM1(1,j)+PWM2(1,3)
%                  end
%                  if myexercise(i,j)=='C'
%                       add(i,1)=add(i,1)+PWM1(2,j)+PWM2(2,3)
%                  end
%                  if myexercise(i,j)=='G'
%                       add(i,1)=add(i,1)+PWM1(3,j)+PWM2(3,3)
%                  end
%                  if myexercise(i,j)=='T'
%                       add(i,1)=add(i,1)+PWM1(4,j)+PWM2(4,3)
%                  end      
%         end
% end