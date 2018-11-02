%Y为按照行展开或者列展开的数据集
function [L] = HyperGraphCal_wu(Y,row,col,K,Datatype,GraphConstructMethod)

M = row;
N = col;
 PixelArea = zeros(row,col);

% 按照行展开计算，给每个像素点标序号
if strcmp(Datatype,'Row_expand')
    for i = 1 : M
        for j = 1:N
            PixelArea(i,j) = (i-1) * N +j;
        end
    end
    
    % 按照列展开计算，给每个像素点标序号
elseif strcmp(Datatype,'Line_expand')
    for i = 1 : N
        for j = 1:M
            PixelArea(j,i) = (i-1) * M +j;
        end
    end
    
else
    error('wrong DATA_TYPE');
end

if strcmp(GraphConstructMethod,'AdaptiveKNN_Global') || strcmp(GraphConstructMethod,'AdaptiveKNN_Local') || strcmp(GraphConstructMethod,'KNN')

    [H,PixelArea,neighbour,sigema] = KNN_Graph_wu(Y,PixelArea,M,N,K,GraphConstructMethod);

elseif strcmp(GraphConstructMethod,'L1Graph_Global')
    usage = 'Cal_H';
    tol = 5e-3;
    Iscompare = 'yes';
    [~,H,sigema,neighbour] = L1Graph(Y,PixelArea,usage,tol,Iscompare);
    %[~,H,sigema,neighbour] = L1Graph(Y,usage,tol);
else
    error('wrong GraphConstructMethod');
end


%计算核参数的系数
sigema = sigema/(sum(neighbour));

w = zeros(1,M*N);
%计算权值W
for i = 1 : M
    for j = 1:N
        if strcmp(GraphConstructMethod,'AdaptiveKNN_Global') || strcmp(GraphConstructMethod,'L1Graph_Global')
            pos_row = i;
            pos_col = j;
        else
            pos_row = i + d;
            pos_col = j + d;            
        end
        pixelpostion = PixelArea(pos_row,pos_col);
        %pixelpostion = (i-1)*N + j;
        %计算位置为（i，j）的像素点对应的超边的权重
        Num_Neighbour = find( H(:, pixelpostion) == 1);
        [Num_Neighbour_size , ~] = size(Num_Neighbour);
        temp = Y(:,Num_Neighbour) - repmat(Y(:,pixelpostion),1,Num_Neighbour_size);
        w(pixelpostion) = 0;
        for k = 1 : Num_Neighbour_size
            w(pixelpostion) = w(pixelpostion) + exp( - (norm(temp(:,k),2))^2 / sigema^2 );
        end
    end
end
% W = diag(w);
W = sparse(1:M*N,1:M*N,w,M*N,M*N);

%计算顶点的度
dv = zeros(1,M*N);
for i = 1 : M*N
    
    %     Num_Vi = find( H(i,:) == 1);
    %     Num_Vi_Size = size(Num_Vi);
    
    dv(i) = w * H(i,:)';
    
end
% Dv = diag(dv);
Dv = sparse(1:M*N,1:M*N,dv,M*N,M*N);

de = zeros(1,M*N);
%计算超边的度
for i = 1 : M*N
    de(i) = sum(H(:,i));
end
% De = diag(de);
De = sparse(1:M*N,1:M*N,de,M*N,M*N);

L = Dv - H*W*(De)^(-1)*H';