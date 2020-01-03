 
load('tzj31.mat')
load('cmd_tzj31.mat')
load('A_tzj31.mat')
t1=ones(280,1);
t2=-ones(560,1);
label=[t1;t2];
h=size(data,1);
selectedfeature=featureSelect(data,A,10511,2000)
res=ones(100,4);

for k=1:100
     N1=0%N1代表正样本的数量
    N2=0%N2代表负样本的数量
    N12=0;%代表判断为负样本的正样本的数量
    N21=0;%代表判断为正样本的负样本的数量
 
    indices = crossvalind('Kfold', h, 5);
    Allaccu=[];
    A=0;
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
%          dataset_scale=dataset
        train_data1=dataset_scale(1:mtrain,:);
        test_data1=dataset_scale((mtrain+1):(mtrain+mtest),:);
        model=svmtrain(train_label,train_data1,cmd);
        [predictlabel,accuracy,decision_values]=svmpredict(test_label,test_data1,model);
        Allaccu(i)=accuracy(1,1);
        str = sprintf( 'Accuracy = %g%% ',accuracy(1,1));
        disp(str);
        sum=0;
        for i=1:length(Allaccu)
        sum=sum+Allaccu(i);
        end
        aveacc=sum/length(Allaccu);
        str2 = sprintf( 'AveAccuracy = %g%% ', aveacc);
        disp(str2);
        for i=1:168
            if test_label(i)==1
              N1=N1+1
            end
            if  test_label(i)==-1
              N2=N2+1
            end
            if  test_label(i)==1 &&  predictlabel(i)==-1
              N12=N12+1
            end
            if test_label(i)==-1 && predictlabel(i)==1
              N21=N21+1
            end
        end
    end
    Sn=1-N12/N1
    Sp=1-N21/N2
    Acc=1-(N12+N21)/(N1+N2)
    Mcc=(1-N12/N1-N21/N2)/sqrt((1+(N21-N12)/N1)*(1+(N12-N21)/N2))
    res(k,1)=Sn;
    res(k,2)=Sp;
    res(k,3)=Mcc;
    res(k,4)=Acc;
end
a=mean(res(:,1))
s=std(res(:,1),0)%i
d=s/sqrt(10)

a=mean(res(:,2))
s=std(res(:,2),0)%i
d=s/sqrt(10)


a=mean(res(:,3))
s=std(res(:,3),0)%i
d=s/sqrt(10)

a=mean(res(:,4))
s=std(res(:,4),0)%i
d=s/sqrt(10)
