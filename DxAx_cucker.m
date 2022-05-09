%This script computes the derivative wrt to x_l of the matrix A(x)
% in the semilinear form for the Cucker Smale model
function [DA] = DxAx_cucker(x,na,nainv,l)
x = x(1:na);
xx = x-x(l);
v = xx./(1+xx.^2).^2;
P = diag(v);
P(l,:) = -v;
P(:,l) = -v;
P(l,l) = sum(v)-v(l);
DA = zeros(2*na); 
DA(na+1:end,na+1:end) = -2*nainv*P;
end