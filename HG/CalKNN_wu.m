function [Num_Final] = CalKNN_wu(Y_input,y_compare,Num,K)
[~,N] = size(Y_input);
for i = 1 : N
    angle(i) = Y_input(:,i)' * y_compare/(norm(Y_input(:,i)) * norm(y_compare) );
    
end
[~ , xulie] = sort(angle,'descend');
Num_Final = Num(xulie);
Num_Final = Num_Final(1:K);
end