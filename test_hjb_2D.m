% Test script for the resolution via TT Gradient Cross of the optimal control problem
% \dot{x_1} = x_2, \dot{x_2} = x_1^3+u
% with cost functional J =  \int q ||x(s)||^2 + gamma |u(s)|^2 ds 
% Input parameters are read from the keyboard

clear
close all

% Gradient weight
lambda = parse_parameter('Parameter lambda (default 1): ', 1);
% Noise amplitude
noise = parse_parameter('Noise amplitude (default 0.01): ', 1e-2);

% Stopping tolerance
tol = 1e-4;
% Matrices and its derivative in the semilinear form of the ODE
Ax_ricc_sl = @(x) [0 1; x(1)^2 0];
gxi = [0;1];

% Derivatice of A(x) wrt to x_1
Ax_ricc_sl_der = @(x) [0 0; 2*x(1) 0];

% State penalty
diag_w = 0.5;
% Control penalty
gamma = 0.5;
gxiGxiGamma = gxi*gxi'/gamma;
% Final time
T = 5;
% Maximum number of iterations
itmax = 10;

% Exact solution for the SDRE and its derivative wrt to x1
P = @ (x) [((x(1)^4 + 1)^(1/2)*(2*(x(1)^4 + 1)^(1/2) + 2*x(1)^2 + 1)^(1/2))/2  (x(1)^4 + 1)^(1/2)/2 + x(1)^2/2;...
    (x(1)^4 + 1)^(1/2)/2 + x(1)^2/2  (2*(x(1)^4 + 1)^(1/2) + 2*x(1)^2 + 1)^(1/2)/2];
dP1 = @(x) [sqrt(2*sqrt(x(1)^4+1)+2*x(1)^2+1)*x(1)^3/sqrt(x(1)^4+1)+sqrt(x(1)^4+1)*(4*x(1)^3/sqrt(x(1)^4+1)+4*x(1))/(4*sqrt(2*sqrt(x(1)^4+1)+2*x(1)^2+1))...
    x(1)^3/sqrt(x(1)^4+1)+x(1); x(1)^3/sqrt(x(1)^4+1)+x(1)  (4*x(1)^3/sqrt(x(1)^4+1)+4*x(1))/(4*sqrt(2*sqrt(x(1)^4+1)+2*x(1)^2+1))];

% Number of basis
n = 14;
% Boundaries of the domain
a = -1;
b = -a;
% Number and sample of initial conditions
N = 10;
x0vec = 2*rand(2,N)-1;
% Grid
[x,w] = lgwt(n,a,b);
x = x';
% Derivative of the bases
Dphi = lagrange_derivative(x,x);
D2phi = Dphi'*Dphi;
y = x;
% TT rank
r = 3;
% SDRE solver
fun2D = @(x,y) fun([x y],Ax_ricc_sl,gxi,diag_w,gamma,Ax_ricc_sl_der,gxiGxiGamma,noise);

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

% TT Gradient Cross

while resid>tol && it<itmax
    % Along the x direction
    for i=1:n
        for j=1:r
            [h(i,j),hx(i,j),hy(i,j)] = fun2D(x(i),y(I2(j)));
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
        for j=1:n
            [h2(i,j),h2x(i,j),h2y(i,j)] = fun2D(x(I1(i)),y(j));
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
        [hxy(i,j),~,~] = fun2D(x(I1(i)),y(I2(j)));
    end
end

nodes = x;
l = @(j,xx) prod(xx-x(1:j-1))*prod(xx-x(j+1:end))/(prod(x(j)-x(1:j-1))*prod(x(j)-x(j+1:end)));
controlfun = @(x) controlfun_gradient(x,nodes,l,gxi,B1,B2',n,gamma,hxy);
odefun = @(x,u) Ax_ricc_sl(x)*x;

t = [0, T];
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);
err = zeros(N,1);
err_u = zeros(N,1);

for k = 1:N
    % Computation of the optimal control and the optimal trajectory
    x0 = x0vec(:,k);
    [t,X_hjb] = ode45(@(t,x) Ax_ricc_sl(x)*x+gxi*controlfun(x) , t, x0, opts);
    ux = zeros(size(t));
    for i = 1:size(X_hjb,1)
        ux(i) = controlfun(X_hjb(i,:)); 
    end
    cost_hjb = sum((X_hjb.^2), 2)*diag_w + gamma*ux.^2;
    total_hjb = sum((t(2:end)-t(1:end-1)).*0.5.*(cost_hjb(1:end-1)+cost_hjb(2:end)));
    
    % Exact solution
    u_exact = @(x) -gxi'*(P(x)*x+0.5*[x'*dP1(x)*x; 0])/gamma;
    [t,X_exact] = ode45(@(t,x) Ax_ricc_sl(x)*x+gxi*u_exact(x) , t, x0, opts);
    ux_exact = zeros(size(t));
    for i = 1:size(X_exact,1)
        ux_exact(i) = u_exact(X_exact(i,:)'); 
    end
    cost_exact = sum((X_exact.^2), 2)*diag_w + gamma*ux_exact.^2;
    total_exact = sum((t(2:end)-t(1:end-1)).*0.5.*(cost_exact(1:end-1)+cost_exact(2:end)));
    err(k) = abs(total_hjb-total_exact);
    err_u(k) = sqrt(sum((t(2:end)-t(1:end-1)).*(ux(1:end-1)-ux_exact(1:end-1)).^2));
end

%% Results

fprintf('Mean error in the cost  = %3.6f\n',sum(err)/N);
fprintf('Mean error in the control %3.6f\n',sum(err_u)/N);

plot(t,ux,t,ux_exact,'Linewidth',2)
grid
title('Optimal control')
legend('TT Gradient Cross','Exact')



function [V,DV1,DV2] = fun(y,Ax_ricc_sl,gxi,diag_w,gamma,Ax_ricc_sl_der,gxiGxiGamma,noise)

n = size(y,1);
for i = 1:n
    x=y(i,:);
    P = icare(Ax_ricc_sl(x),gxi,diag_w,gamma,'noscaling');
    P = P+noise*rand(2);
    A1 = Ax_ricc_sl(x)'-P*gxiGxiGamma;
    A2 = Ax_ricc_sl(x)-gxiGxiGamma*P;
    dP1 = sylvester(A1,A2,-(Ax_ricc_sl_der(x)'*P+P*Ax_ricc_sl_der(x)));
    dP1 = dP1+noise*rand(size(dP1));
    DV = P*x';
    V = x*DV;
    DV1 = 2*DV(1)+x*dP1*x';
    DV2 = 2*DV(2);
end

end


function [u] = controlfun_gradient(xnew,nodes,l,gxi,B1,B2,n,gamma,hxy)

Dphi = lagrange_derivative(xnew,nodes);
Phi1 = zeros(1,n);
Phi2 = Phi1;
for j = 2:n-1
    Phi1(j) = l(j,xnew(1));
    Phi2(j) = l(j,xnew(2));
end
Phi1(1) = prod(xnew(1)-nodes(2:end))/prod(nodes(1)-nodes(2:end));
Phi1(n) = prod(xnew(1)-nodes(1:end-1))/prod(nodes(end)-nodes(1:end-1));
Phi2(1) = prod(xnew(2)-nodes(2:end))/prod(nodes(1)-nodes(2:end));
Phi2(n) = prod(xnew(2)-nodes(1:end-1))/prod(nodes(end)-nodes(1:end-1));
DV1 = Dphi(1,:)*B1*hxy*B2*Phi2';
DV2 = Phi1*B1*hxy*B2*Dphi(2,:)';
u = -(DV1*gxi(1)+DV2*gxi(2))/(2*gamma);

end