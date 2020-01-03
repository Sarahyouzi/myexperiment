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
    r1=size(myexercise,1);



A=zeros(r1,1)%A�ĺ���
C=zeros(r1,1)%C�ĺ���
G=zeros(r1,1)%G�ĺ���
T=zeros(r1,1)%T�ĺ���
len=zeros(r1,1)%ÿ�����еĳ���
AT=zeros(r1,1)% 1 A+T�ĺ���
A_T=zeros(r1,1)%A-T�ĺ���
GC=zeros(r1,1)% 2 G+C�ĺ���
G_C=zeros(r1,1)%G-C�ĺ���
GdC=zeros(r1,1)% 3 G/C
gc=zeros(r1,1)% 4 G-C/G+C
at=zeros(r1,1)% 5 A-T/A+T
ac=zeros(r1,1)% 6 A+T/G+C
for i=1:r1%r1��������������
    len(i,1)=size(deblank(myexercise(i,:)),2);
    for j=1:len(i,1)%r2��AGTC����������ϵ�����
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

