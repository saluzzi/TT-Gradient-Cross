function [Dphi] = lagrange_derivative(x,nodes)
%Derivates of the lagrangian basis computed 
%on the nodes and evaluted in x
n=length(x);
M=length(nodes);
Dphi=zeros(n,M);
for i=1:n
    for j=1:M
        sum=0;
        for k=1:M
            if k~=j
                product=1;
                for m=1:M
                    if m~=k && m~=j
                        product=product*(x(i)-nodes(m))/(nodes(j)-nodes(m));
                    end
                end
                sum=sum+product/(nodes(j)-nodes(k));
            end
        end
        Dphi(i,j)=sum;
    end
end

end