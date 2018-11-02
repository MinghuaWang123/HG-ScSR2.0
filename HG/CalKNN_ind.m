function [Num_Final] = CalKNN_ind(Y_input,y_compare,Num,K)
[~,N] = size(Y_input);
for i = 1 : N
    angle(i) = Y_input(:,i)' * y_compare/(norm(Y_input(:,i)) * norm(y_compare) );
    
end
[~ , xulie] = sort(angle,'descend');

Num_Final = Num(xulie);
r=find(0<Num_Final & Num_Final<=(N/3));
g=find((N/3)<Num_Final & Num_Final<=(2*N/3));
b=find((2*N/3)<Num_Final & Num_Final<=N);
Num_Final2 = [Num_Final(r(1:K)) ; Num_Final(g(1:K)) ; Num_Final(b(1:K))];
Num_Final = sort(Num_Final2);
end