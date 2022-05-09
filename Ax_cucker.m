%This script computes the matrix A(x) in the semilinear form for the
%Cucker Smale model

function [A] = Ax_cucker(x,na,nainv)
x = x(1:na);
P = nainv./(1+(x-x').^2);
P = P-diag(sum(P));
A = [zeros(na) eye(na); zeros(na) P];
end