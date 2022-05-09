function [ind,C] = maxvol_rect(A,K)
% Input:
%   A: a n x r matrix
%   K: the total number of sought indices (must be >=r)
% Output:
%   ind: a vectors of K indices
%   C: the "interpolation" matrix

[n,r]=size(A);
if (r>=n)||(K>=n)
    ind = 1:n;
    C = eye(n);
    return;
end
% Select initial r indices
ind = maxvol2(A);
% Initial coeff matrix
C = A/A(ind,:);

% We will nullify this vector at the selected indices
chosen = ones(n,1);
chosen(ind) = 0;
% compute square 2-norms of each row in matrix C
row_norm2 = sum(repmat(chosen, 1, r).*C.*conj(C), 2);
% Find the heaviest row
[~,imax]=max(row_norm2);
% Greedy loop
for k=r+1:K
    % Add the last max index
    ind(k) = imax;
    chosen(imax) = 0;
    % The chosen row
    c = C(imax,:);
    % The expansion vector is C*c'
    v = C*c';
    % Rescaling by the maximal element
    l = 1/(1+v(imax));
    % Correct the coefficients
    C = C-l*v*c;
    % Expand the coeffs
    C = [C, l*v];                                                      %#ok
    % Subtract the norms
    row_norm2 = row_norm2 - real(l*v.*conj(v));
    row_norm2 = row_norm2.*chosen;
    % Next maximal index
    [~,imax] = max(row_norm2);
end

end
