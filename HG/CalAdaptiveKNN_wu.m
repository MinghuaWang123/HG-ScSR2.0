% 此函数计算近邻的时候并不一定满足K近邻要求，可以动态变化
% 只要满足一定的阈值要求即表示具有近邻关系
% 若找不到这样的近邻，则采用KNN算法得到其K个近邻y_compare = y_compare / norm(y_compare,2);
function Num_Final = CalAdaptiveKNN_wu(Y_Database,y_compare,Num,K,yy)
[Dim,N] = size(Y_Database);

%首先对数据进行归一化：即令 ||y||_2 = 1
norm_Y = sqrt(sum(Y_Database.^2));
Y_Database = Y_Database./(repmat(norm_Y,Dim,1));
y_compare  =Y_Database(:,yy);

norm_Database = norm( repmat(y_compare,1,N) -  Y_Database , 'fro')^2;
% kernel_result = [];
for i = 1: N
    kernel = norm(y_compare - Y_Database(:,i) , 'fro')^2 /  norm_Database;
    kernel_result(i) = exp(-kernel);
end

yuzhi = 1/N * sum(kernel_result);
% xulie = find(kernel_result > yuzhi);
Num_Final = Num(kernel_result > yuzhi);
% [m1 n1]   = size(Num_Final);
% 
% if m1 > K
%   Num_Final = Num_Final(1:K);
% end
    
[Num_Final_Size,~] = size(Num_Final);
% 若找不到这样的近邻，则采用KNN算法得到其K个近邻
if Num_Final_Size < 2
    [Num_Final] = CalKNN_wu(Y_Database,y_compare,Num,K);
end
end