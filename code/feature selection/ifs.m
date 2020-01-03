
function [record] = ifs(A,data,cmd)
%data为输入的特征数据集
%A为使用二项分布排出的顺序
%result为使用IFS进项特征选择之后得出的对应的准确率
F=[];
features=size(data,2)
record=ones(features,2)
%将数据按照使用二项分布得出的顺序进行排序
for j=1:features
    E=data(:,A(j));  
    F=[F E];
end
sortdata=F;
F1=[];
for j=6786:features
    j
    record(j,1)=j
    E=sortdata(:,j);
    F1=[F1 E];
    selectedfeature=F1    
    sum=0;
t1=ones(280,1);
t2=-ones(560,1);
label=[t1;t2];
h=size(data,1);
res=ones(2,1);
for k=1:2
    indices = crossvalind('Kfold', h, 5);
    Allaccu=[];
    for i =1:5
        testdata=(indices == i);
        traindata=~testdata;
        test_data=selectedfeature(testdata,:);
        train_data=selectedfeature(traindata,:);

        testlabel=(indices == i);
        trainlabel=~testlabel;
        test_label=label(testlabel,:);
        train_label=label(trainlabel,:);

        [mtrain,ntrain]=size(train_data);
        [mtest,ntest]=size(test_data);
        dataset=[train_data;test_data];
        [dataset_scale,ps]=mapminmax(dataset',-1,1);
        dataset_scale=dataset_scale';
        train_data1=dataset_scale(1:mtrain,:);
        test_data1=dataset_scale((mtrain+1):(mtrain+mtest),:);
        model=svmtrain(train_label,train_data1,cmd);
        [predictlabel,accuracy,decision_values]=svmpredict(test_label,test_data1,model);
        Allaccu(i)=accuracy(1,1);
        sum=0;
        for i=1:length(Allaccu)
        sum=sum+Allaccu(i);
        end
        aveacc=sum/length(Allaccu);
    end
    res(k)=aveacc;
end
        record(j,2)=mean(res(:))
end
 

% figure
%  x=1:1:features;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
%  yvalue=record(:,2)
%  plot(x,yvalue,'-b','linewidth',2); %线性，颜色，标记
% axis([0,features,0,100])  %确定x轴与y轴框图大小
% set(gca,'XTick',[0:128:features]) %x轴范围1-6，间隔1
% set(gca,'YTick',[0:5:100]) %y轴范围0-700，间隔100
% xlabel('Mumber of Pentamers')  %x轴坐标描述
% ylabel('Overall accuracy(%)') %y轴坐标描述
% %  text(x(226),yvalue(226),['(',num2str(x(226)),',',nu  m2str(yvalue(226)),')']);
%  [max_a,index]=max(yvalue)
end