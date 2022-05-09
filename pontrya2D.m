function [V,dV1,dV2] = pontrya2D(x,gamma,solinit,F,dF,ccostfun)
% Pontryagin solver for the 2D system 
% \dot{x_1} = x_2, \dot{x_2} = x_1^3+u

BVP_bc2=@(ya,yb) BVP_bc(ya,yb,x);
fjac2=@(x,y) fjac(y,dF);
options = bvpset('RelTol',1e-12,'BCJacobian',@bcjac,'FJacobian',fjac2);
BVP_ode=@(t,x) BVP_ode2(x,gamma,F);
sol = bvp4c(BVP_ode,BVP_bc2, solinit, options);
t = sol.x;
y = sol.y;
ut=F(-0.5*y(4,:)/gamma);

cost=0.5*sum(y(1:2,:).^2)+ccostfun(ut);
V=sum((t(2:end)-t(1:end-1)).*0.5.*(cost(1:end-1)+cost(2:end)));
dV1=y(3,1);
dV2=y(4,1);

end

%------------------------------------------------
% Boundary value problem
function dydt = BVP_ode2(y,gamma,F)

u =  F(-0.5*y(4)/gamma);
dydt=zeros(1,4);
dydt(1) = y(2);
dydt(2) = y(1)^3+u;
dydt(3) = -y(1)-3*y(1)^2*y(4);
dydt(4) =-y(2)-y(3);

end

% Jacobian of the system
function  dfdy = fjac(y,dF)

n=length(y);
dfdy=zeros(n);
dfdy(1:2,1:2)=[0 1; 3*y(1)^2 0];
dfdy(2,4)=dF(y(4));
dfdy(3:4,:)=[-1-6*y(1)*y(4) 0 0 -3*y(1)^2; 0 -1 -1 0];

end
% -----------------------------------------------
% Boundary conditions

function res = BVP_bc(ya,yb,x0)

d=length(x0);
res = [ ya(1:d)-x0'
    yb(d+1:2*d)-0];

end

% Jacobian boundary conditions
function  [dbcdya,dbcdyb] = bcjac(~,~)

dbcdya=zeros(4);
dbcdya(1:2,1:2)=eye(2);
dbcdyb=zeros(4);
dbcdyb(3:4,3:4)=eye(2);

end
