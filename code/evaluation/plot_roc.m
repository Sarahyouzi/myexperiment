function  auc = plot_roc( predict, ground_truth )  
% INPUTS  
%  predict       - 分类器对测试集的分类结果  
%  ground_truth - 测试集的正确标签,这里只考虑二分类，即0和1  
% OUTPUTS  
%  auc            - 返回ROC曲线的曲线下的面积  
predict=result_23(:,2)
ground_truth=result_23(:,1)
predict=result(:,2)
ground_truth=result(:,1)
%初始点为（1.0, 1.0）  
x = 1.0;  
y = 1.0;  
%计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num  
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);  
%根据该数目可以计算出沿x轴或者y轴的步长  
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  
%首先对predict中的分类器输出值按照从小到大排列  
[predict,index] = sort(predict);  
ground_truth = ground_truth(index);  
%对predict中的每个样本分别判断他们是FP或者是TP  
%遍历ground_truth的元素，  
%若ground_truth[i]=1,则TP减少了1，往y轴方向下降y_step  
%若ground_truth[i]=0,则FP减少了1，往x轴方向下降x_step  
for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        y = y - y_step;  
    else  
        x = x - x_step;  
    end  
    X(i)=x;  
    Y(i)=y;  
end  
%画出图像       
plot(X1,Y2,'-ro','LineWidth',2,'MarkerSize',3);  
hold on
plot(X,Y,'-ro','LineWidth',2,'MarkerSize',3);  
box off
ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on
xlabel('Specifity');  
ylabel('Sensitivity');  
title('ROC曲线图');  
%计算小矩形的面积,返回auc  
auc = -trapz(X,Y);   
auc = -trapz(X1,Y); 
end  