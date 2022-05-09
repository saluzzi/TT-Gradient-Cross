% Test script for the resolution via TT Gradient Cross of the optimal control problem
% of the Cucker-Smale system
% with cost functional J =  \int  ||x(s)||^2 / na + |u(s)|^2 / na ds 
% Input parameters are read from the keyboard

clear
close all

% Gradient weight
lambda = parse_parameter('Parameter lambda (default 1e-3): ', 1e-3);
% Number of agents
na = parse_parameter('Number of agents (d/2) (default 2): ', 2);
% Two Boxes approach
TB = parse_parameter('You want to apply the Two Boxes approach? (default 0) ', 0);
if TB
   aTB = parse_parameter('Size of the second box (default 0.02): ', 0.02);
end
nainv = 1/na;
% Computation domain [-av,av]^d
av = 0.5;
% Dimension of the problem
d = 2*na;
% Number of basis
nq = 5;
% Actuator
gxi = [zeros(na); eye(na)];
% Cost parameters
gamma = 1/na;
wxi = gamma;
% Stopping tolerance
tol_stop = 1e-2;
% TT truncation tolerance
tol_trunc = 1.e-5;
% Final time
T = 10;
% No control constraints
umax = inf;
% equispaced initial condition in [a,b]
a = 0;
b = 0.4;
% Maximum number of sweeps for the TT Cross
nswp = 6;
% Matrix in the semilinear form
Ax_ricc_sl = @(x) Ax_cucker(x, na, nainv);
% Generate Gauss points and weights
[x1v,~] = lgwt(nq, -av, av);
[P,dP] = legendre_rec(x1v, -av, av, numel(x1v)-1);
W = gxi*gxi'/gamma;
nainv = 1/na;
repmatP = repmat({cat(d,P,dP)},d,1);
fun_V = @(x) fun(x, Ax_ricc_sl, gxi, wxi, gamma, W, na, nainv, reshape(reshape(1:d,2,[])', 1, []));

% TT Gradient Cross
V_sl = gradient_cross(repmat({x1v},d,1), repmatP, fun_V, tol_trunc, lambda, 'nswp',nswp,'tol_exit',tol_stop);
V_sl = round(V_sl, tol_trunc);
%[V_sl,evals,ranks] = hjb_cucker(d, nv, av, gxi, wxi, gamma, tol_stop, tol_trunc,lambda,nswp, 1);
V = core2cell(V_sl); % Disintegrate tt_tensor class to speed up evaluations
% Ordering the variables in pair x_1,v_1,...x_n,v_n
gfun = @(x,i,j) repmat(double(j==2*i),size(x,1),1);  % Actuator 
ipermind = reshape(reshape(1:d,[],2)', 1, []); 

% State cost function
lfun = @(x) sum((x.^2).*(wxi'), 2);
% Feedback control based on the TT tensor V
controlfun = @(x) multicontrolfun_leg(x(:,ipermind), -av, av, V, na, gfun, gamma);
if TB
    % Two Boxes approach
    [K]=lqr(Ax_ricc_sl(zeros(1,na)), gxi, wxi, gamma);
    controlfun = @(x) (max(abs(x)) <= aTB)*(-K*x')'+ (max(abs(x)) > aTB)*controlfun(x);
end
odefun = @(x) x*Ax_cucker(x, na, nainv)';  % Dynamics

% Initial state in space
x0 = linspace(a,b,d+1);
x0 = x0(1:end-1);
% Set up integrator options
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);

% Create a handle for the control cost
if (isinf(umax))
    ccostfun = @(u) gamma*sum(u.^2,2);
else
    ccostfun = @(u) (2*gamma) * (umax.*u.*atanh(min(max(u/umax,-1),1)) + umax.^2*log(max(1-u.^2/umax.^2,0))/2);
end

% Solve HJB-controlled ODE
[t,X_hjb] = ode45(@(t,x)odefun(x').'+gxi*controlfun(x').' , [0 T], x0', opts);
ux = zeros(length(t), na);
for i=1:size(X_hjb,1)
    ux(i,:) = controlfun(X_hjb(i,:)); % Save the control separately
end
cost_hjb = sum((X_hjb.^2).*(wxi'), 2) + ccostfun(ux);
total_hjb = sum((t(2:end)-t(1:end-1)).*0.5.*(cost_hjb(1:end-1)+cost_hjb(2:end)));

%% Computation of the optimal trajectory directly via SDRE

n=length(t);
nainv=1/na;
X=zeros(d,n);
X(:,1)=x0';

% Control computed via SDRE
control_sdre2=@(x)  (-lqr(Ax_ricc_sl(x),gxi,wxi,gamma)*x')';
[t_sdre,X_sdre] = ode45(@(t,x)odefun(x').'+gxi*control_sdre2(x').' , t, x0', opts);
u_sdre = zeros(length(t_sdre), na);
for i=1:size(X_sdre,1)
    u_sdre(i,:) = control_sdre2(X_sdre(i,:));
end
cost_sdre = sum((X_sdre.^2).*(wxi'),2) +ccostfun(u_sdre);
total_sdre=  sum((t_sdre(2:end)-t_sdre(1:end-1)).*0.5.*(cost_sdre(1:end-1)+cost_sdre(2:end)));


%% Results

fprintf('total SDRE cost = %3.6f\n', total_sdre);
fprintf('total TT cost = %3.6f\n', total_hjb);
error=abs(total_hjb-total_sdre);
fprintf('Error TT-SDRE= %3.6f\n', error);
error_relative=abs(total_hjb-total_sdre)/abs(total_sdre);
fprintf('Relative Error TT-SDRE= %3.6f\n', error_relative);
maximum=max(abs(X_hjb(end,:)));
fprintf('Max dynamics= %3.6f\n', maximum);
plot(t,X_hjb,'LineWidth',2)
grid
title('Optimal trajectory')


% SDRE solver

function V=fun(x, Ax_ricc_sl, gxi, diag_w, gamma, W, na, nainv, permind)

[m,n] = size(x);
V = zeros(m,n+1);
ipermind(permind) = 1:n;
for i = 1:m
    y = x(i,permind);
    Ay = Ax_ricc_sl(y);
    P = icare(Ay,gxi,diag_w,gamma,'noscaling');
    v = (2*P*y')';
    v = v(ipermind);
    v = [y*P*y', v];
    A = Ay'-P*W;
    B = Ay-W*P;
    for j = 1:na
        Dx = DxAx_cucker(y,na,nainv,j);
        X = sylvester(A,B,-(Dx'*P+P*Dx));
        v(ipermind(j)+1) = v(ipermind(j)+1)+y*X*y';
    end
    V(i,:) = v;
end

end
