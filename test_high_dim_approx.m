% Test script for the approximation of high-dimensional functions 
% via TT Gradient Cross
% Input parameters are read from the keyboard

clear
close all

d = 100;
try
    % Grid points
    [x,w] = lgwt(33,-1,1);
    % Basis functions and theri derivatives
    [P,dP] = legendre_rec(x,-1,1,numel(x)-1);
catch ME
    error('No Legendre functions in the path, please add TT-HJB');
end

choice = parse_parameter(['Choose the function f(x) (default 1): \n 1) f(x)= exp(-sum(x)/(2d))' ...
    '\n 2) f(x)= exp(-prod(x))  \n >> ' ], 1);
% Gradient weight
lambda = parse_parameter('Parameter lambda (default 0.01): ', 1e-2);

switch choice
    case 1
        fun = @(x,noise) fun_rank_1(x,noise);
        % Load of the Amen Cross approximation obtained with tolerance
        % 1e-12
        load('yex_fun_rank_1')
        % Rank increment per iteration 
        kickrank = 0;
    case 2
        fun = @(x,noise) fun_rank_not_1(x,noise);
         % Load of the Amen Cross approximation obtained with tolerance
        % 1e-12
        load('yex_fun_rank_not_1')
        % Rank increment per iteration 
        kickrank = 1;
end

% Stopping tolerance for the TT Gradient Cross
tol = 1e-4;
% Noise amplitudes
noise_vec = [0 10.^(-6:-1)];
num = length(noise_vec);
err = zeros(1,num);
% Cell array of vectors of grid points
repx = repmat({x},d,1);
% Cell array of basis functions and their derivatives
repP = repmat({cat(d,P,dP)},d,1);
% TT matrix of the basis functions
mkronP = mtkron([{tt_matrix(P)} repmat({tt_matrix(P)},1,d-1)]) ;

for k = 1:num
    noise = noise_vec(k);
    % Approximation via TT Gradient Cross
    y = gradient_cross(repx, repP, @(x) fun(x,noise), tol, lambda, 'y0',1,'kickrank', kickrank);
    % Computation of the error
    err(k) = norm(mkronP*y-yex)/norm(yex);
end

% Plots of the approximation error varying the noise amplitudes
if num == 7
    fig = semilogy(err,'LineWidth',2);
    xticklabels({'0','1e-6','1e-5','1e-4','1e-3','1e-2','1e-1'})
    grid
    xlabel('Noise amplitude')
    title('Approximation error')
    ax = fig.Parent;
    set(ax, 'YTick', 10.^(-14:2:6))
end

function F =fun_rank_1(x,noise)

[m,n]=size(x);
F=zeros(m,n+1);
c=1/(2*m);
for i=1:m
    y=x(i,:);
    F(i,1)=exp(-sum(y)*c);
    for j=1:n
        F(i,j+1)=-c*F(i,1)*exp(-y(j)*c);
    end
end
F=F+noise*randn(m,n+1);

end

function F =fun_rank_not_1(x,noise)

[m,n]=size(x);
F=zeros(m,n+1);
for i=1:m
    v=zeros(1,n+1);
    y=x(i,:);
    v(1)=exp(-prod(y));
    v(2)=-v(1)*prod(y(2:end));
    for j=2:n-1
        v(j+1)=-v(1)*prod(y(1:j-1))*prod(y(j+1:end));
    end
    v(n+1)=-v(1)*prod(y(1:end-1));
    F(i,:)=v;
end
F=F+noise*randn(m,n+1);

end