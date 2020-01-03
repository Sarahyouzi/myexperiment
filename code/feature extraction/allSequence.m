function [ss] = allSequence(k)
%此函数为统计k联核苷酸的出现频率
% kinds为传过来的某序列中k联核苷酸的出现情况
s1='ACGT';
n=4^k;
ss=cell(n,1);
u=1;
if k==1
    for i=1:4
             s=s1(i);
             ss{u}=s;
             u=u+1;
       
    end   
end
if k==2
    for i=1:4
        for j=1:4
             s=strcat(s1(i),s1(j));
             ss{u}=s;
             u=u+1;
        end
    end   
end
if k==4
    for i=1:4
        for j=1:4
            for k=1:4
                for l=1:4
                        s=strcat(s1(i),s1(j),s1(k),s1(l));
                            ss{u}=s;
                            u=u+1;
                end
            end
        end
    end
end
if k==3
    for i=1:4
        for j=1:4
            for k=1:4
                s=strcat(s1(i),s1(j),s1(k));
                ss{u}=s;
                u=u+1;
            end
        end
    end
end
if k==5
    for i=1:4
        for j=1:4
            for k=1:4
                for l=1:4
                    for m=1:4
                        s=strcat(s1(i),s1(j),s1(k),s1(l),s1(m));
                            ss{u}=s;
                            u=u+1;
                    end
                end
            end
        end
    end
end
if k==6
    for i=1:4
        for j=1:4
            for k=1:4
                for l=1:4
                    for m=1:4
                        for n=1:4
                            s=strcat(s1(i),s1(j),s1(k),s1(l),s1(m),s1(n));
                            ss{u}=s;
                            u=u+1;
                        end
                    end
                end
            end
        end
    end
end
end

