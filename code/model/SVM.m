%4、给样本数据添加标签
l1=ones(280,1)
l2=zeros(560,1)
l3=[l1;l2]
data1=[l3 data]

%6、根据推荐的特征组合，来整理特征集
selectedfeature=featureSelect(data,A,10050,6483)  
data2=[selectedfeature l3]
[data21,ps] = mapminmax(selectedfeature');
data21=data21'

data21=[data21 l3]
%7、使用svm
% 1. 随机产生训练集和测试集
%%
% 2. 训练集——640个样本
train_matrix = selectedfeature;
train_label = l3;

%%
% 3. 测试集——200个样本


%% III. 数据归一化
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';

%% IV. SVM创建/训练(RBF核函数)
%%
% 1. 寻找最佳c/g参数——交叉验证方法
[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
%第二步改变范围
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
         %-v参数，说明返回的不是model而是一个标量数
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);%把最好得到找出来
            bestc = 2^c(i,j);%同时记录c和g的值
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            %当已经找到最佳，且满足精度要求，为了加快速度，那么以c为标准取一个
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
% 2. 创建/训练SVM模型
model = svmtrain(train_label,Train_matrix,cmd);

%% V. SVM仿真测试
[predict_label_1,accuracy_1,dec_value] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,dec_value] = svmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];

%% VI. 绘图
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)