function [H,PixelArea,neighbour,sigema] = KNN_Graph_wu(Y,PixelArea,M,N,K,GraphConstructMethod)
H = sparse(M*N , M*N);

sigema = 0;
neighbour = [];

    %�����ͨ��ȫ�ֱȽϼ���ͼ���ڽӵ�
    for i = 1 : M
        for j = 1:N
            %(i-1)*N + j
            %ԭ��λ��Ϊ��i��j�������ص㣬���ڵ�λ��Ϊ��pos_row��pos_col��
            pos_row = i;
            pos_col = j;
           
            Num = unique(PixelArea(:));
        
            LocalY = Y(:,Num);
            
            %�Ƚϵ�
            y_compare = Y(:,PixelArea(pos_row,pos_col));
            yy        = PixelArea(pos_row,pos_col);
            %����ȫ�ֽ��ڵ�
            [Num_Neighbour] = CalAdaptiveKNN_wu(LocalY,y_compare,Num,K,yy);
           
            
            [Num_Neighbour_size,~] = size(Num_Neighbour);
            %����ͼ����,ÿ�����߰����Ķ���
            H( Num_Neighbour , PixelArea(pos_row,pos_col) ) = 1;
            
            
            %����˲���
            temp = Y(: , Num_Neighbour) - repmat(y_compare,1,Num_Neighbour_size);
            temp = temp.^(2);
            sigema = sigema + sum(sqrt(sum(temp,1)));
            %�����Ӧ�Ľ��ڵ�������洢��neighbour
            neighbour = [neighbour,Num_Neighbour_size];
        end
    end
