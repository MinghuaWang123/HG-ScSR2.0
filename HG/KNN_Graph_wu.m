function [H,PixelArea,neighbour,sigema] = KNN_Graph_wu(Y,PixelArea,M,N,K,GraphConstructMethod)
H = sparse(M*N , M*N);

sigema = 0;
neighbour = [];

    %如果是通过全局比较计算图的邻接点
    for i = 1 : M
        for j = 1:N
            %(i-1)*N + j
            %原来位置为（i，j）的像素点，现在的位置为（pos_row，pos_col）
            pos_row = i;
            pos_col = j;
           
            Num = unique(PixelArea(:));
        
            LocalY = Y(:,Num);
            
            %比较点
            y_compare = Y(:,PixelArea(pos_row,pos_col));
            yy        = PixelArea(pos_row,pos_col);
            %计算全局近邻点
            [Num_Neighbour] = CalAdaptiveKNN_wu(LocalY,y_compare,Num,K,yy);
           
            
            [Num_Neighbour_size,~] = size(Num_Neighbour);
            %计算图矩阵,每个超边包含的顶点
            H( Num_Neighbour , PixelArea(pos_row,pos_col) ) = 1;
            
            
            %计算核参数
            temp = Y(: , Num_Neighbour) - repmat(y_compare,1,Num_Neighbour_size);
            temp = temp.^(2);
            sigema = sigema + sum(sqrt(sum(temp,1)));
            %计算对应的近邻点个数，存储于neighbour
            neighbour = [neighbour,Num_Neighbour_size];
        end
    end
