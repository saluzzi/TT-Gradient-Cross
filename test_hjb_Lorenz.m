% Test script for the resolution via TT Gradient Cross of the optimal control problem
% of the Lorenz system
% with cost functional J =  \int  ||x(s)||^2 + gamma |u(s)|^2 ds 
% Input parameters are read from the keyboard

clear
close all

% Gradient weight
lambda = parse_parameter('Parameter lambda (default 1): ', 1);
% Control penalty
gamma = parse_parameter('Parameter lambda (default 0.001): ', 0.001);
% Stopping tolerance
tol_stop = 1.e-2;
% TT Truncation tolerance 
tol_trunc = 1.e-5;
% Initial condition
x0 = [-1 -1 -1];

% Final time for the simulation
T = 5;
% No control constraint
umax = inf;
% Parameters of the system
d = 3;
sigma = 10;
beta = 8/3;
ro = 2;
diag_w = 1;
% Matrices in the semilinear form and their derivatives
Ax = @(x) [-sigma sigma 0; ro-x(3) -1 0; x(2) 0 -beta];
DAx2 = [0 0 0; 0 0 0; 1 0 0];
DAx3 = [0 0 0; -1 0 0; 0 0 0];
gxi = [0; 1; 0];
W = gxi*gxi'/gamma;
% Number of basis
nq = 6;
% Computation domain [-av,av]^d
av = 1;
% Grid
[x1v,w1v] = lgwt(nq, -av, av);
% Legendre basis and theier derivatives
[P,dP] = legendre_rec(x1v, -av, av, numel(x1v)-1);
PWd = tt_matrix(repmat({diag(w1v)*P},d,1));
repmatP = repmat({cat(d,P,dP)},d,1);
% SDRE solver
fun2 = @(x) fun(x, Ax, gxi, diag_w, gamma, W, DAx2, DAx3);

% TT Gradient Cross
V_sl = gradient_cross(repmat({x1v},d,1), repmatP, fun2, tol_stop, lambda);

V = core2cell(V_sl); % Disintegrate tt_tensor class to speed up evaluations
gfun = @(x,l,i) gxi(i,:); % Actuator (here just const vector)
controlfun = @(x) multicontrolfun_leg(x, -av, av, V, 1, gfun, gamma);
%controlfun = @(x) controlfun_leg_lorenz(x, -av, av, V, gfun, gamma, umax);
odefun = @(x) x*Ax(x)';
% Set up integrator options
t = [0, T];
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);
ccostfun = @(u) gamma*sum(u.^2,2);

% Computation of the optimal control and the optimal trajectory
[t,X_hjb] = ode45(@(t,x) odefun(x').'+gxi*controlfun(x').' , t, x0', opts);
ux=zeros(size(t));
for i=1:size(X_hjb,1)
    ux(i) = controlfun(X_hjb(i,:)); % Save the control separately
end
cost_hjb = sum((X_hjb.^2).*(diag_w'), 2) + ccostfun(ux);
total_hjb = sum((t(2:end)-t(1:end-1)).*0.5.*(cost_hjb(1:end-1)+cost_hjb(2:end)));

% Results

fprintf('Total cost = %3.6f\n', total_hjb);
plot(t,X_hjb(:,1),t,X_hjb(:,2),t,X_hjb(:,3),'LineWidth',2)
legend('x1','x2','x3')
grid
title('Optimal trajectory')
figure
plot(t,ux,'LineWidth',2)
title('Optimal Control')
grid


% SDRE solver

function V=fun(x,Ax,gxi,diag_w,gamma,W,DAx2,DAx3)
[m,n]=size(x);
V=zeros(m,n+1);
for i=1:m
    y=x(i,:);
    Ay=Ax(y);
    P=icare(Ay,gxi,diag_w,gamma,'noscaling');
    V(i,:)=[y*P*y', (2*P*y')'];
    A=Ay'-P*W;
    B=Ay-W*P;
    X=sylvester(A,B,-(DAx2'*P+P*DAx2));
    V(i,3)=V(i,3)+y*X*y';
    X=sylvester(A,B,-(DAx3'*P+P*DAx3));
    V(i,4)=V(i,4)+y*X*y';
end
end
