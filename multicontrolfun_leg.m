% Multivariate Control function from TT -- Legendre Galerkin
% Inputs: 
%   x: points where u and V need to be interpolated
%   a,b: left and right ends of the state interval
%   V: a cell array of TT cores of the value function, containing
%      coefficients in the basis of normalised Legendre polynomials on [a,b]
%   m: dimension of the control space
%   gfun: a @(x,i,j)function returning the (i,j)-th element of the actuator
%         matrix evaluated at state x. i should be from 1 to m (the index
%         of the control component), and j should be from 1 to d (the index
%         of the grad V).
%   gamma: control regularisation parameter
%
% Outputs:
%   ux: a nt x m array of control signal evaluations
%   Vx: a nt x 1 vector of value function
function [ux,Vx] = multicontrolfun_leg(x,a,b,V,m,gfun,gamma)
d = numel(V);
nv = size(V{1},2);
nt = size(x,1);
ux = zeros(nt,m);
if (nargin>1)
    Vx = zeros(nt,1);
end
[p,dp] = legendre_rec(x(:), a, b, nv-1);
p = reshape(p, nt, d, nv);
dp = reshape(dp, nt, d, nv);
for k=1:nt 
    % compute (g grad) V
    for j=1:d
        Vxj = 1;
        for i=1:d
            % Interpolate jth component of grad V
            Vxj = Vxj*reshape(V{i}, size(Vxj,2), []);
            Vxj = reshape(Vxj, nv, []);
            if (i==j)
                % Differentiate in the jth variable
                Vxj = reshape(dp(k,i,:), 1, [])*Vxj;
            else
                Vxj = reshape(p(k,i,:), 1, [])*Vxj;
            end
        end
        % Assemble control components
        for ell=1:m
            gli = gfun(x(k,:), ell, j);
            if (~isempty(gli))
                ux(k,ell) = ux(k,ell) - Vxj*gli/(2*gamma);
            end
        end
    end
    if (nargin>1)
        % Interpolate V
        Vxj = 1;
        for i=1:d
            Vxj = Vxj*reshape(V{i}, size(Vxj,2), []);
            Vxj = reshape(Vxj, nv, []);
            Vxj = reshape(p(k,i,:), 1, [])*Vxj;
        end
        Vx(k) = Vxj;
    end
end
end

