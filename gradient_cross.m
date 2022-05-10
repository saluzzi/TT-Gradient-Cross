% Inputs:
%   Grids: a d x 1 cell array of vectors of grid points
%   Bases: a d x 1 cell array of basis functions and their derivatives
%          evaluated on Grids. i-th cell should be an nx x ny x 2 array of
%          the form cat(3, Phi, dPhi), where nx is the number of points in 
%          Grids{i}, ny is the number of basis functions in the i-th
%          variable, and Phi, dPhi are nx x ny matrices of function and
%          derivative evaluations respectively.
%   soughtfun: a @(X)function handle that takes M x d matrix of coordinates
%              and returns a M x (d+1) matrix of sought function & gradient
%              evaluations
%   tol: truncation tolerance
%   lambda: gradient regularisation parameter
% varargins can be given in pairs
%   underfactor: a factor of attempted undersampling in the least squares
%                problem (1 means no undersampling) !!! CAUTION !!! too
%                small a value may make the problem unstable! Reasonable
%                values range between 0.6 and 1.                        [1]
%   kickrank: rank increment per iteration (using random values)        [4]                                 
%   nswp: maximal number of iterations                                 [20]
%   y0: initial rank                                                    [4]
%   verb: verbosity level, 0-silent, 1-sweep info, 2-block info         [1]
%   vec: whether funs can accept and return vectorized values        [true]
%   tol_exit: stopping tolerance                                      [tol]
%   stop_sweep: number of extra sweeps done after err<tol is hit        [0]
%   exitdir: if 1, return after the forward sweep, if -1, return the
%            backward sweep, if 0, after any                            [0]
%   dir: direction of the first computing sweep
%        The warm-up goes in the opposite direction                     [1]
%
% Output: a d-dimensional tt_tensor of coefficients with mode sizes ny,
%         and the total number of soughtfun evaluations

function [y, evalcnt]=gradient_cross(Grids, Bases, soughtfun, tol, lambda, varargin)

y = 4; % means 4 random indices at start
nswp = 20;
kickrank = 4;
underfactor = 1; % The factor of attempted undersampling
verb = 1;
vec = true;
exitdir=0;
tol_exit = tol;
stop_sweep = 0;
dir = 1;

i = 1;
vars = varargin;
while (i<length(vars))
    switch lower(vars{i})
        case 'y0'
            y=vars{i+1};
        case 'nswp'
            nswp=vars{i+1};
        case 'stop_sweep'
            stop_sweep=vars{i+1};
        case 'kickrank'
            kickrank=vars{i+1};
        case 'underfactor'
            underfactor=vars{i+1};            
        case 'verb'
            verb = vars{i+1};
        case 'vec'
            vec = vars{i+1};
        case 'exitdir'
            exitdir=vars{i+1};
        case 'tol_exit'
            tol_exit=vars{i+1};
        case 'dir'
            dir=vars{i+1};
        otherwise
            warning('Option %s was not recognized', vars{i});
    end
    i=i+2;
end

d = numel(Grids);
tol_local = tol/sqrt(d);
% Grid sizes
nx = cellfun(@numel, Grids);

if (numel(lambda)==1)
    lambda = repmat(lambda, 1, d);
end

% Choose direction where we start warm-up iteration
dir = -dir;
istart = d;
if (dir>0)
    istart = 1;
end

% Chunks of solution TT cores with gradients sampled on cross points
% First and last matrices are dummies for convenience
SampledCores = cell(d+1,1);
SampledCores{1} = ones(1,1,1); SampledCores{d+1} = ones(1,1,1);
% Each SampledCore = cat(3, y(X), d_1 y(X), ..., d_i y(X))

% Residual cores sampled at residual indices
SampledZZ = cell(d+1,1);
SampledZZ{1} = ones(1,1,1); SampledZZ{d+1} = ones(1,1,1);
% Solution cores sampled at residual indices
SampledZY = cell(d+1,1);
SampledZY{1} = ones(1,1,1); SampledZY{d+1} = ones(1,1,1);

% Initial random samples and TT storage
X = cell(d+1,1); X{1} = zeros(1,0); X{d+1} = zeros(1,0);
ry = [1; y*ones(d-1,1); 1];
% Residual samples
Xz = cell(d+1,1); Xz{1} = zeros(1,0); Xz{d+1} = zeros(1,0);
rz = [1; kickrank*ones(d-1,1); 1];
n = ones(d,1);
y = cell(d,1);
z = cell(d,1);
i = istart;
while (i~=(d-istart+1))
    iprev = (1-dir)/2; % where is the previous rank relative to i
    inext = (1+dir)/2; % where is the next rank relative to i
    n(i) = size(Bases{i},2);  % basis is nx(i) x n(i) x 2
    % Random initial guess for coefficients
    y{i} = randn(ry(i), n(i), ry(i+1));
    % Maxvol and get discrete indices
    [y{i},ry(i+inext),ind] = maxvol_core(y{i},dir,Bases{i},SampledCores{i+iprev}(:,:,1),[],[]);
    % Split index into rank and physical subindices
    indsep = tt_ind2sub([ry(i+iprev), nx(i)], ind(:));
    % Subsample coordinates
    if (dir>0)
        X{i+inext} = [X{i+iprev}(indsep(:,1), :), Grids{i}(indsep(:,2))];
    else
        X{i+inext} = [Grids{i}(indsep(:,2)), X{i+iprev}(indsep(:,1), :)];
    end    
    % Sample core and its derivatives
    SampledCores{i+inext} = resample_gradients(SampledCores{i+iprev},ind,dir,y{i},Bases{i});
    
    if (kickrank>0)
        % Sample residual
        z{i} = randn(rz(i), n(i), rz(i+1));
        % Maxvol and get discrete indices
        [z{i},rz(i+inext),ind] = maxvol_core(z{i},dir,Bases{i},SampledZZ{i+iprev}(:,:,1),[],[]);
        % Split index into rank and physical subindices
        indsep = tt_ind2sub([rz(i+iprev), nx(i)], ind(:));
        % Subsample coordinates
        if (dir>0)
            Xz{i+inext} = [Xz{i+iprev}(indsep(:,1), :), Grids{i}(indsep(:,2))];
        else
            Xz{i+inext} = [Grids{i}(indsep(:,2)), Xz{i+iprev}(indsep(:,1), :)];
        end
        % Sample residual core and its derivatives
        SampledZZ{i+inext} = resample_gradients(SampledZZ{i+iprev},ind,dir,z{i},Bases{i});
        % Sample solution core at residual indices
        SampledZY{i+inext} = resample_gradients(SampledZY{i+iprev},ind,dir,y{i},Bases{i});
    end
    
    i = i+dir;
end
n(d-istart+1) = size(Bases{d-istart+1},2);
y{d-istart+1} = randn(ry(d-istart+1), n(d-istart+1), ry(d-istart+1+1));
z{d-istart+1} = randn(rz(d-istart+1), n(d-istart+1), rz(d-istart+1+1));


% Start the computation loop
swp = 1;
dir = -dir;
istart = d-istart+1;
i = istart;
iprev = (1-dir)/2; % where is the previous rank relative to i
inext = (1+dir)/2; % where is the next rank relative to i
last_swp = 0;
max_dy = 0;
evalcnt = 0;
undersampled = 0;
while (swp<=nswp)
    % Sample function and solve LS problem on coefficients
    [cry,evalcnt,undersampled] = solve_core(i,SampledCores{i},Bases{i},SampledCores{i+1},lambda,underfactor, X{i},X{i+1},Grids{i},soughtfun,vec,evalcnt,undersampled, []);
    
    % Estimate the error in the coefficients
    y{i} = reshape(y{i}, ry(i)*n(i)*ry(i+1), 1);
    dy = max(abs(cry-y{i}))/max(abs(cry));
    max_dy = max(max_dy, dy);
    
    % Switch to the next block
    cry = reshape(cry, ry(i), n(i), ry(i+1), 1);
    y{i} = cry;
    if (i~=(d-istart+1))
        % Truncate the core before computing enrichment
        [y{i},ry(i+inext),y{i+dir},yapprox] = truncate_core(cry,dir,y{i+dir},tol_local);
        EnrichCore = [];
        err = nan;
        if (kickrank>0)
            % Compute residuals
            yz = assemble_sampled_tt(SampledZY{i},Bases{i},SampledZY{i+1},yapprox);
            [crz,evalcnt,undersampled] = solve_core(i,SampledZZ{i},Bases{i},SampledZZ{i+1},lambda,1, Xz{i},Xz{i+1},Grids{i},soughtfun,vec,evalcnt,undersampled, yz);
            crz = reshape(crz, rz(i), n(i), rz(i+1), 1);
            err = norm(crz(:))/norm(yz(:));
            if (dir>0)
                yz = assemble_sampled_tt(SampledCores{i},Bases{i},SampledZY{i+1},yapprox);
                [EnrichCore,evalcnt,undersampled] = solve_core(i,SampledCores{i},Bases{i},SampledZZ{i+1},lambda,1, X{i},Xz{i+1},Grids{i},soughtfun,vec,evalcnt,undersampled, yz);
                EnrichCore = reshape(EnrichCore, ry(i), n(i), rz(i+1), 1);
            else
                yz = assemble_sampled_tt(SampledZY{i},Bases{i},SampledCores{i+1},yapprox);
                [EnrichCore,evalcnt,undersampled] = solve_core(i,SampledZZ{i},Bases{i},SampledCores{i+1},lambda,1, Xz{i},X{i+1},Grids{i},soughtfun,vec,evalcnt,undersampled, yz);
                EnrichCore = reshape(EnrichCore, rz(i), n(i), ry(i+1), 1);
            end
        end
        % Truncate, QR, Maxvol and get discrete indices
        [y{i},ry(i+inext),ind,y{i+dir}] = maxvol_core(y{i},dir,Bases{i},SampledCores{i+iprev}(:,:,1),y{i+dir},EnrichCore);
        % Split index into rank and physical subindices
        indsep = tt_ind2sub([ry(i+iprev), nx(i)], ind(:));
        % Subsample coordinates
        if (dir>0)
            X{i+inext} = [X{i+iprev}(indsep(:,1), :), Grids{i}(indsep(:,2))];
        else
            X{i+inext} = [Grids{i}(indsep(:,2)), X{i+iprev}(indsep(:,1), :)];
        end
        % Sample core and its derivatives
        SampledCores{i+inext} = resample_gradients(SampledCores{i+iprev},ind,dir,y{i},Bases{i});
        
        if (kickrank>0)
            % Maxvol for the residual
            [z{i},rz(i+inext),ind,z{i+dir}] = maxvol_core(crz,dir,Bases{i},SampledZZ{i+iprev}(:,:,1),z{i+dir},[]);
            % Split index into rank and physical subindices
            indsep = tt_ind2sub([rz(i+iprev), nx(i)], ind(:));
            % Subsample coordinates
            if (dir>0)
                Xz{i+inext} = [Xz{i+iprev}(indsep(:,1), :), Grids{i}(indsep(:,2))];
            else
                Xz{i+inext} = [Grids{i}(indsep(:,2)), Xz{i+iprev}(indsep(:,1), :)];
            end
            % Sample core and its derivatives
            SampledZZ{i+inext} = resample_gradients(SampledZZ{i+iprev},ind,dir,z{i},Bases{i});
            SampledZY{i+inext} = resample_gradients(SampledZY{i+iprev},ind,dir,y{i},Bases{i});
        end
    end
    
    if (verb>1)
        fprintf('\t-gradient_cross- swp=%d, i=%d, dy=%3.3e, ranks=[%d,%d], n=%d, |z|=%3.3e\n', swp, i, dy, ry(i), ry(i+1), n(i), err);
    end
    
    
    i = i+dir;
    % Change direction, check for exit
    if (i==(d-istart+1+dir))
        if (verb>0)
            fprintf('=gradient_cross= swp=%d, max_dy=%3.3e, max_rank=%d, max_n=%d, cum#evals=%d, cum#reduced=%d\n', swp, max_dy, max(ry), max(n), evalcnt, undersampled);
        end
        if (max_dy<tol_exit)
            last_swp = last_swp+1;
        end
        if ((last_swp>stop_sweep)||(swp>=nswp))&&((dir==exitdir)||(exitdir==0))
            break;
        end
        dir = -dir;
        istart = d-istart+1;
        iprev = (1-dir)/2; % where is the previous rank relative to i
        inext = (1+dir)/2; % where is the next rank relative to i        
        swp = swp+1;
        max_dy = 0;
        i = i+dir;
    end
end

y = cell2core(tt_tensor, y);
end

% Truncate current core
function [ycur,rnext,ynext,yapprox] = truncate_core(ycur,dir,ynext,tol)
% By default assume ycur=y{i}, ynext = y{i+1} (direction +)
if (dir<0)
    ycur = permute(ycur, [3,2,1]);
    if (~isempty(ynext))
        ynext = permute(ynext, [3,2,1]);
    end
end
[rprev,ncur,rnext]=size(ycur);
if (~isempty(ynext))
    [~,nnext,rother]=size(ynext);
end
ycur = reshape(ycur, rprev*ncur, rnext);

Rt = 1;
rnextorig = rnext;
if (~isempty(tol))&&(tol>0)
    % Full-pivot cross truncation to satisfy C-norm threshold
    [ycur,Rt] = localcross(ycur, tol);
    rnext = size(ycur,2);
end
yapprox = ycur*Rt;

ycur = reshape(ycur, rprev, ncur, rnext);
yapprox = reshape(yapprox, rprev, ncur, rnextorig);

% Cast non-orth factor to get initial guess in the next iteration
if (~isempty(ynext))
    ynext = reshape(ynext, [], nnext*rother);
    ynext = Rt*ynext;
    ynext = reshape(ynext, [], nnext, rother);
end
% Permute the dimensions back
if (dir<0)
    ycur = permute(ycur, [3,2,1]);
    yapprox = permute(yapprox, [3,2,1]);
    if (~isempty(ynext))
        ynext = permute(ynext, [3,2,1]);
    end
end
end

% Enrich, QR and maxvol current core
function [ycur,rnext,ind,ynext] = maxvol_core(ycur,dir,Phi,SampledPrevCore,ynext,EnrichCore)
% By default assume ycur=y{i}, ynext = y{i+1} (direction +)
if (dir<0)
    ycur = permute(ycur, [3,2,1]);
    if (~isempty(ynext))
        ynext = permute(ynext, [3,2,1]);
    end
    if (~isempty(EnrichCore))
        EnrichCore = permute(EnrichCore, [3,2,1]);
    end    
end
[rprev,ncur,rnext]=size(ycur);
if (~isempty(ynext))
    [~,nnext,rother]=size(ynext);
end
ycur = reshape(ycur, rprev*ncur, rnext);

if (~isempty(EnrichCore))
    % Enrichment (if we have one)
    EnrichCore = reshape(EnrichCore, rprev*ncur, []);
%     EnrichCore = randn(rprev*ncur, size(EnrichCore,2));
    ycur = [ycur, EnrichCore];
end
% QR in case if we have an enrichment
[ycur,Rq] = qr(ycur, 0);
% Aggregate factors from truncation and enrichment
Rq = Rq(:, 1:rnext);
rnext = size(ycur,2);

% Cast coefficients to Point evaluations
pcur = reshape(ycur, rprev, ncur*rnext);
pcur = SampledPrevCore*pcur;
rprevs = size(SampledPrevCore, 1); % rprev of the samples - can be different
pcur = pcur.';
pcur = reshape(pcur, ncur, rnext*rprevs);
pcur = Phi(:,:,1)*pcur;
nxcur = size(Phi,1);
pcur = reshape(pcur, nxcur*rnext, rprevs);
pcur = pcur.';
pcur = reshape(pcur, rprevs*nxcur, rnext);

% Maxvol and divide
ind = maxvol2(pcur);
YY = pcur(ind,:);
ycur = ycur/YY;       % Now Phi_<(X_<) x Phi_i(X_i) x ycur = identity
ycur = reshape(ycur, rprev, ncur, rnext);

% Cast non-orth factors to get initial guess in the next iteration
if (~isempty(ynext))
    Rq = YY*Rq;
    ynext = reshape(ynext, [], nnext*rother);
    ynext = Rq*ynext;
    ynext = reshape(ynext, [], nnext, rother);
end
% Permute the dimensions back
if (dir<0)
    ycur = permute(ycur, [3,2,1]);
    if (~isempty(ynext))
        ynext = permute(ynext, [3,2,1]);
    end
end
end


% Sample current core of gradients
function [SampledNext] = resample_gradients(SampledPrev,ind,dir,ycur,Phi)
% By default assume ycur=y{i}, ynext = y{i+1} (direction +)
if (dir<0)
    ycur = permute(ycur, [3,2,1]);
end
[rprev,ncur,rnext]=size(ycur);

% each SampledCore is of size r x r x (i+1, dir>0; d-i+1, dir<0)
d_prev = size(SampledPrev, 3);
rprevs = size(SampledPrev, 1); % rprev of the samples - can be different
rnexts = numel(ind); % can be oversampled too
SampledNext = zeros(rnexts, rnext, d_prev+1);

% Expand previous funcs/gradients with zero-order function in current variable
nxcur = size(Phi,1);
for i=1:d_prev
    gcur = reshape(ycur, rprev, ncur*rnext);
    gcur = SampledPrev(:,:,i)*gcur;
    gcur = gcur.';
    gcur = reshape(gcur, ncur, rnext*rprevs);
    gcur = Phi(:,:,1)*gcur;
    gcur = reshape(gcur, nxcur*rnext, rprevs);
    gcur = gcur.';
    gcur = reshape(gcur, rprevs*nxcur, rnext);
    SampledNext(:,:,i) = gcur(ind, :);
end

% Derivative in the current variable
gcur = reshape(ycur, rprev, ncur*rnext);
gcur = SampledPrev(:,:,1)*gcur;
gcur = gcur.';
gcur = reshape(gcur, ncur, rnext*rprevs);
gcur = Phi(:,:,2)*gcur;
gcur = reshape(gcur, nxcur*rnext, rprevs);
gcur = gcur.';
gcur = reshape(gcur, rprevs*nxcur, rnext);
SampledNext(:,:,d_prev+1) = gcur(ind, :);
end



% % Evaluate samples of the function
% function [crf,evalcnt] = evaluate_fun(i,X,grid,soughtfun,vec,evalcnt)
% Samples = indexmerge(X{i}, grid, X{i+1});
% if (vec)
%     crf = soughtfun(Samples);
%     % Check if the user function is sane
%     if (size(crf,1)~=size(Samples,1))
%         error('%d samples requested, but %d values received. Check your function or use vec=false', size(Samples,1), size(crf,1));
%     end    
% else
%     % We need to vectorize the fun
%     crf = soughtfun(Samples(1,:));
%     b = numel(crf);
%     crf = reshape(crf, 1, b);
%     crf = [crf; zeros(size(Samples,1)-1, b)];
%     for j=2:size(Samples,1)
%         crf(j,:) = soughtfun(Samples(j,:));
%     end
% end
% evalcnt = evalcnt + size(Samples,1);
% end

% Merges two or three indices in the little-endian manner
function [J]=indexmerge(varargin)
sz1 = max(size(varargin{1},1),1);
sz2 = max(size(varargin{2},1),1);
sz3 = 1;
if (nargin>2) % Currently allows only 3
    sz3 = max(size(varargin{3}, 1), 1);
end
% J1 goes to the fastest index, just copy it
J1 = repmat(varargin{1}, sz2*sz3, 1);
% J2 goes to the middle
J2 = reshape(varargin{2}, 1, []);
J2 = repmat(J2, sz1, 1); % now sz1 ones will be the fastest
J2 = reshape(J2, sz1*sz2, []);
J2 = repmat(J2, sz3, 1);
J = [J1,J2];
if (nargin>2)
    % J3 goes to the slowest
    J3 = reshape(varargin{3}, 1, []);
    J3 = repmat(J3, sz1*sz2, 1); % now sz1 ones will be the fastest
    J3 = reshape(J3, sz1*sz2*sz3, []);
    J = [J,J3];
end
end


% Eval and Solve the least-squares problem on the core coefficients
function [cry,evalcnt,undersampled] = solve_core(i,SampledPrev,Phi,SampledNext,lambda,underfactor, Xprev,Xnext,grid,soughtfun,vec,evalcnt,undersampled,CoreToSubtract)
d = size(SampledPrev,3) + size(SampledNext, 3) - 1;
rprev = size(SampledPrev,2);
ncur = size(Phi,2);
rnext = size(SampledNext,2);

% Maxvol to search redundant points in prev coordinates
Aleft = reshape(SampledPrev, rprev*rprev, i);
Aleft = Aleft .* [1 sqrt(lambda(1:i-1))];
Aleft = Aleft.';
Aleft = reshape(Aleft, i*rprev, rprev);
indleft = maxvol_rect(Aleft, i*rprev*underfactor);
ind_sep = tt_ind2sub([i, rprev], indleft(:));
[indleft_u, ~, insertleft] = unique(ind_sep(:,2), 'stable');
labelleft = ind_sep(:,1);
% insert will replicate function values to ind, which selects matrix rows
Aleft = Aleft(indleft, :);
Mleft = SampledPrev(indleft_u,:,1);

% Maxvol to search redundant points in the current coordinate
A = reshape(Phi, ncur*ncur, 2);
A = A .* [1 sqrt(lambda(i))];
A = A.';
A = reshape(A, 2*ncur, ncur);
ind = maxvol_rect(A, 2*ncur*underfactor);
ind_sep = tt_ind2sub([2, ncur], ind(:));
[ind_u, ~, insert] = unique(ind_sep(:,2), 'stable');
label = ind_sep(:,1);
% insert will replicate function values to ind, which selects matrix rows
A = A(ind, :);
M = Phi(ind_u,:,1);

% Maxvol to search redundant points in next coordinates
Aright = reshape(SampledNext, rnext*rnext, d-i+1);
Aright = Aright .* [1 sqrt(lambda(d:-1:i+1))];
Aright = Aright.';
Aright = reshape(Aright, (d-i+1)*rnext, rnext);
indright = maxvol_rect(Aright, (d-i+1)*rnext*underfactor);
ind_sep = tt_ind2sub([d-i+1, rnext], indright(:));
[indright_u, ~, insertright] = unique(ind_sep(:,2), 'stable');
labelright = ind_sep(:,1);
% insert will replicate function values to ind, which selects matrix rows
Aright = Aright(indright, :);
Mright = SampledNext(indright_u,:,1);

% Actual sampling set
Samples = indexmerge(Xprev(indleft_u,:), grid(ind_u), Xnext(indright_u,:));
evalcnt = evalcnt + size(Samples,1);
undersampled = undersampled + rprev*ncur*rnext-size(Samples,1);

% Sample the function
if (vec)
    crf = soughtfun(Samples);
    % Check if the user function is sane
    if (size(crf,1)~=size(Samples,1))
        error('%d samples requested, but %d values received. Check your function or use vec=false', size(Samples,1), size(crf,1));
    end
else
    % We need to vectorize the fun
    crf = soughtfun(Samples(1,:));
    b = numel(crf);
    crf = reshape(crf, 1, b);
    crf = [crf; zeros(size(Samples,1)-1, b)];
    for j=2:size(Samples,1)
        crf(j,:) = soughtfun(Samples(j,:));
    end
end
% If we are computing the residual, subtract the current approximation
if (~isempty(CoreToSubtract))
    crf = crf - CoreToSubtract;
end

% Duplicate point values where necessary, but sample corr. gradient parts
crf = reshape(crf, numel(indleft_u), numel(ind_u), numel(indright_u), d+1);
crf1 = zeros(numel(indleft), numel(ind_u), numel(indright_u), 1);
for j=1:numel(indleft)    
    if (labelleft(j)>1)                      % labelleft is OK going 1 to i
        crf1(j,:,:) = crf(insertleft(j),:,:, labelleft(j)) * sqrt(lambda(labelleft(j)-1));
    else
        crf1(j,:,:) = crf(insertleft(j),:,:, 1);
    end
end
crf2 = zeros(numel(indleft_u), numel(ind), numel(indright_u), 1);
for j=1:numel(ind)
    if (label(j)>1)         % label is only 1 or 2 but we need i grad component
        crf2(:,j,:) = crf(:,insert(j),:, i+1) * sqrt(lambda(i));
    else
        crf2(:,j,:) = crf(:,insert(j),:, 1);
    end
end
crf3 = zeros(numel(indleft_u), numel(ind_u), numel(indright), 1);
for j=1:numel(indright)
    if (labelright(j)>1)
        crf3(:,:,j) = crf(:,:,insertright(j), d-labelright(j)+3) * sqrt(lambda(d-labelright(j)+2));
    else
        crf3(:,:,j) = crf(:,:,insertright(j), 1);
    end
end

% RHS of the normal equations
crf = rank1mult(Aleft', M', Mright', crf1) ...
    + rank1mult(Mleft', A', Mright', crf2) ...
    + rank1mult(Mleft', M', Aright', crf3);

% Gram matrices
Mleft = Mleft'*Mleft;
Aleft = Aleft'*Aleft;
M = M'*M;
A = A'*A;
Mright = Mright'*Mright;
Aright = Aright'*Aright;

% A-matrices should be "fairly" conditioned due to maxvol, but M can be
% rank-deficient due to undersampling

% fprintf('conds: [%g %g] x [%g %g] x [%g %g]\n', cond(Mleft), cond(Aleft), cond(M), cond(A), cond(Mright), cond(Aright));

% cry = pcg(@(x)rank1mult(Aleft,M,Mright,x)+rank1mult(Mleft,A,Mright,x)+rank1mult(Mleft,M,Aright,x), ...
%           crf, 1e-8, 1e4);

% Generalised diagonalisation
[Vleft,Lleft] = eig(Mleft,Aleft,'chol');  Lleft = reshape(diag(Lleft), [],1,1);
[V,L] = eig(M,A,'chol');  L = reshape(diag(L), 1,[],1);
[Vright,Lright] = eig(Mright,Aright,'chol');   Lright = reshape(diag(Lright), 1,1,[]);

% full matrix = A*V*(L x L x I + ...)*V'*A
cry = rank1solve(Aleft,A,Aright, crf);
cry = rank1solve(Vleft,V,Vright, cry);

% eigs of 3D matrix
Lall = L.*Lright + Lleft.*Lright + Lleft.*L;
Lall = Lall(:);
cry = cry./Lall;

cry = rank1solve(Vleft.',V.',Vright.', cry);
cry = rank1solve(Aleft,A,Aright, cry);
end

% Multiply vec(3D tensor) by matrices in all modes, Y = X x_1 A1 x_2 A2 x_3 A3
function [y] = rank1mult(A1,A2,A3, x)
n1 = size(A1,2); n2 = size(A2,2); n3 = size(A3,2);
y = reshape(x, n1, []); % rem size is n2*n3
y = A1*y;
y = y.';
y = reshape(y, n2, []); % rem size is n3*m1
y = A2*y;
y = y.';
y = reshape(y, n3, []); % rem size is m1*m2
y = A3*y;
y = y.';
y = y(:);
end

% Multiply vec(3D tensor) by inv(matrices) in all modes, 
% Y = X x_1 A1^{-1} x_2 A2^{-1} x_3 A3^{-1}
function [y] = rank1solve(A1,A2,A3, x)
n1 = size(A1,1); n2 = size(A2,1); n3 = size(A3,1);
y = reshape(x, n1, []); % rem size is n2*n3
y = A1\y;
y = y.';
y = reshape(y, n2, []); % rem size is n3*m1
y = A2\y;
y = y.';
y = reshape(y, n3, []); % rem size is m1*m2
y = A3\y;
y = y.';
y = y(:);
end


% Assemble a sampled TT format from sampled left and right interfaces, and
% the current core
function [z] = assemble_sampled_tt(SampledPrev,Phi,SampledNext,y)
rprev = size(SampledPrev,2);
ncur = size(Phi,2);
rnext = size(SampledNext,2);
d = size(SampledPrev,3) + size(SampledNext, 3) - 1;

% zero-order term first
zi = reshape(y, rprev, []);
zi = SampledPrev(:,:,1)*zi;
zi = zi.';
zi = reshape(zi, ncur, []);
zi = Phi(:,:,1)*zi;
zi = zi.';
zi = reshape(zi, rnext, []);
zi = SampledNext(:,:,1)*zi;
zi = zi.';
zi = zi(:);
z = zeros(numel(zi), d+1);
z(:,1) = zi;

% Left derivatives
i = 1;
while (i<size(SampledPrev,3))
    zi = reshape(y, rprev, []);
    zi = SampledPrev(:,:,i+1)*zi;
    zi = zi.';
    zi = reshape(zi, ncur, []);
    zi = Phi(:,:,1)*zi;
    zi = zi.';
    zi = reshape(zi, rnext, []);
    zi = SampledNext(:,:,1)*zi;
    zi = zi.';
    z(:,i+1) = zi(:);
    i = i+1;
end
% current derivative
zi = reshape(y, rprev, []);
zi = SampledPrev(:,:,1)*zi;
zi = zi.';
zi = reshape(zi, ncur, []);
zi = Phi(:,:,2)*zi;
zi = zi.';
zi = reshape(zi, rnext, []);
zi = SampledNext(:,:,1)*zi;
zi = zi.';
z(:,i+1) = zi(:);
i = i+1;
% right derivatives
while (i<=d)
    zi = reshape(y, rprev, []);
    zi = SampledPrev(:,:,1)*zi;
    zi = zi.';
    zi = reshape(zi, ncur, []);
    zi = Phi(:,:,1)*zi;
    zi = zi.';
    zi = reshape(zi, rnext, []);
    zi = SampledNext(:,:,d-i+2)*zi;
    zi = zi.';
    z(:,i+1) = zi(:);
    i = i+1;    
end


end
