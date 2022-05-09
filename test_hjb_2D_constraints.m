% Test script for the resolution via TT Gradient Cross of the optimal
% control problem with control constraints
% \dot{x_1} = x_2, \dot{x_2} = x_1^3+u
% with cost functional J=  \int q ||x(s)||^2 +  W(u(s)) ds
% Input parameters are read from the keyboard

clear
close all

% Gradient weights
lambda = parse_parameter('Parameter lambda (default 0.01): ', 0.01);
% Control constraint
umax = parse_parameter('Control constraint (default 25): ', 25);
% Initial condition
x0 = [2; 2];
% Stopping tolerance
tol = 1e-4;
% TT rank
r = 6;
% Matrices in the semilinear form of the ODE
Ax_ricc_sl = @(x) [0 1; x(1)^2 0];
gxi = [0;1];

% State penalty
diag_w = 0.5;
% Control penalty
gamma = 0.5;
% Final time
T = 10;
% Maximum number of iterations
itmax = 10;

% Number of basis
n = 14;
% Boundaries of the domain
a = -2;
b = -a;
% Grid
[x,w] = lgwt(n,a,b);
x = x';
% Derivative of the bases
Dphi = lagrange_derivative(x,x);
D2phi = Dphi'*Dphi;
y = x;
% Imposing the control constraint
if umax == inf
    F = @(u) u;
    dF = @(u) -0.5/gamma;
    ccostfun = @(u) gamma*u.^2;
else
    F = @(u) umax*tanh(u/umax);
    dF=@(x) -0.5*sech(0.5*x/(gamma*umax))^2/gamma;
    ccostfun = @(u) (2*gamma) * (umax.*u.*atanh(min(max(u/umax,-1),1)) + umax.^2*log(max(1-u.^2/umax.^2,0))/2);
end

% Initial guess for Pontryagin solver
solinit = bvpinit(linspace(0,T,100),[0 0 1 1]);
% Pontryagin solver
fun2D = @(x,y) pontrya2D([x y],gamma,solinit,F,dF,ccostfun);

% Terms in the Lyapunov equation
Ax1 = D2phi*lambda+eye(n);
Ay2 = Ax1;
% Initial guess
C0 = randn(n,r);
[H,~] = qr(C0,0);
[I2,B2] = maxvol3(H);

hx_prev = zeros(n,r);
hy_prev = zeros(n,r);
resid = inf;
h=  zeros(n,r);
hx = h;
hy = h;
h2 = zeros(r,n);
h2x = h2;
h2y = h2;
it = 0;

%% TT Gradient Cross

while resid>tol && it<itmax
    % Along the x direction
    yy=y(I2);
    for i=1:n
        xx=x(i);
        for j=1:r
            [h(i,j),hx(i,j),hy(i,j)] = fun2D(xx,yy(j));
        end
    end
    appx = Dphi(I2,:)*B2;
    Ax2 = lambda*(appx'*appx);
    Ax3 = h+Dphi'*hx*lambda+hy*appx*lambda;
    R = lyap(Ax1,Ax2,-Ax3);
    % We compute the increment in the function evaluations
    resid = norm(R-hx_prev, 'fro')/norm(R, 'fro');
    [H,R2] = qr(R,0);
    [I1,B1] = maxvol3(H);
    % B1 is the ultimate result of the X-part of the current iteration
    hx_prev = B1;
    % The approximate low-rank model now is B1 * H(I1,:) * R * hy_prev',
    % whereas the exact interpolation reads B1 * hy'.
    % Multiply all but B1 in the approximate model together to compare it
    % to the exact hy next.
    hy_prev = hy_prev*(H(I1,:)*R2).';

    % Along the y direction
    for i=1:r
        xx=x(I1(i));
        for j=1:n
            [h2(i,j),h2x(i,j),h2y(i,j)] = fun2D(xx,y(j));
        end
    end
    appy = Dphi(I1,:)*B1;
    Ay1 = lambda*(appy'*appy);
    Ay3 = h2+lambda*h2y*Dphi+appy'*h2x*lambda;
    C = lyap(Ay1,Ay2,-Ay3);
    % By now hy_prev has absorved all modifications we made after hx
    resid = max(resid, norm(C'-hy_prev, 'fro')/norm(C', 'fro'));
    [H,R] = qr(C',0);
    [I2,B2] = maxvol3(H);
    hy_prev = B2;
    % The approximate low-rank model is now [hx_prev * R' * H(I2,:)'] * B2'
    hx_prev = hx_prev*(H(I2,:)*R).';
    it = it+1;
end
hxy = zeros(r);

for i = 1:r
    for j = 1:r
        hxy(i,j) = fun2D(x(I1(i)),y(I2(j)));
    end
end

% Computation of the optimal control and the optimal trajectory

nodes = x;
l = @(j,xx) prod(xx-x(1:j-1))*prod(xx-x(j+1:end))/(prod(x(j)-x(1:j-1))*prod(x(j)-x(j+1:end)));
controlfun = @(x) controlfun_gradient(x,nodes,l,gxi,B1,B2',n,gamma,hxy,F);
odefun = @(x,u) Ax_ricc_sl(x)*x;

t = [0, T];
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);

[t,X_hjb] = ode45(@(t,x) Ax_ricc_sl(x)*x+gxi*controlfun(x) , t, x0, opts);
ux = zeros(size(t));
for i = 1:size(X_hjb,1)
    ux(i) = controlfun(X_hjb(i,:));
end
cost_hjb = sum((X_hjb.^2), 2)*diag_w + ccostfun(ux);
total_hjb = sum((t(2:end)-t(1:end-1)).*0.5.*(cost_hjb(1:end-1)+cost_hjb(2:end)));

%% Result

plot(t,ux,'Linewidth',2)
grid
title('Optimal control')
figure
plot(t,X_hjb,'Linewidth',2)
grid
title('Optimal trajectory')
fprintf('Total cost  = %3.6f\n',total_hjb);

function [u] = controlfun_gradient(xnew,nodes,l,gxi,B1,B2,n,gamma,hxy,F)

Dphi=lagrange_derivative(xnew,nodes);
Phi1=zeros(1,n);
Phi2=Phi1;
for j=2:n-1
    Phi1(j)=l(j,xnew(1));
    Phi2(j)=l(j,xnew(2));
end
Phi1(1)=prod(xnew(1)-nodes(2:end))/prod(nodes(1)-nodes(2:end));
Phi1(n)=prod(xnew(1)-nodes(1:end-1))/prod(nodes(end)-nodes(1:end-1));
Phi2(1)=prod(xnew(2)-nodes(2:end))/prod(nodes(1)-nodes(2:end));
Phi2(n)=prod(xnew(2)-nodes(1:end-1))/prod(nodes(end)-nodes(1:end-1));
DV1=Dphi(1,:)*B1*hxy*B2*Phi2';
DV2=Phi1*B1*hxy*B2*Dphi(2,:)';
u=-(DV1*gxi(1)+DV2*gxi(2))/(2*gamma);
u=F(u);

end