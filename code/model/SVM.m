%4��������������ӱ�ǩ
l1=ones(280,1)
l2=zeros(560,1)
l3=[l1;l2]
data1=[l3 data]

%6�������Ƽ���������ϣ�������������
selectedfeature=featureSelect(data,A,10050,6483)  
data2=[selectedfeature l3]
[data21,ps] = mapminmax(selectedfeature');
data21=data21'

data21=[data21 l3]
%7��ʹ��svm
% 1. �������ѵ�����Ͳ��Լ�
%%
% 2. ѵ��������640������
train_matrix = selectedfeature;
train_label = l3;

%%
% 3. ���Լ�����200������


%% III. ���ݹ�һ��
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';

%% IV. SVM����/ѵ��(RBF�˺���)
%%
% 1. Ѱ�����c/g��������������֤����
[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
%�ڶ����ı䷶Χ
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    i
    for j = 1:n
        j
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
         %-v������˵�����صĲ���model����һ��������
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);%����õõ��ҳ���
            bestc = 2^c(i,j);%ͬʱ��¼c��g��ֵ
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            %���Ѿ��ҵ���ѣ������㾫��Ҫ��Ϊ�˼ӿ��ٶȣ���ô��cΪ��׼ȡһ��
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
save('cmd_tzj14.mat','cmd')
record=ifs(A,data,cmd)
save('record_tzj14','record')
dot=find(record(:,2)==max(record(:,2)));
[Y_col,ind_row]=max(record)
%%
% 2. ����/ѵ��SVMģ��
model = svmtrain(train_label,Train_matrix,cmd);

%% V. SVM�������
[predict_label_1,accuracy_1,dec_value] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,dec_value] = svmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];

%% VI. ��ͼ
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�SVMԤ�����Ա�(RBF�˺���)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)