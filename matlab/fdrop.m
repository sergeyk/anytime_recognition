function [W, train_err, test_err] = fdrop(X, Y, p, T, lambda, debug, test_X, test_Y)
% X : d x n
% Y: K x n
% p: maximum number of features that will be removed at test time
% T: scalar iteration
[K, n] = size(Y);
X = [X; ones(1, n)];
d = size(X, 1);
W = zeros(K, d);

if exist('test_X', 'var') && exist('test_Y', 'var') && ~isempty(test_X) && ~isempty(test_Y)
    test_X = [test_X; ones(1, length(test_Y))];
    trueTestY = [1:K]*test_Y;
end

trueY = [1:K]*Y;
Fregion = sqrt(2*mean(trueY ~= 1)/lambda);
for t = 1:T
    idx = mod(t, n)+1;
    yy = trueY(idx);
    %yy = [1:K]*Y(:, idx);
    xx = X(:, idx);
    wxy = W(yy, :)'.*xx;
    [~,I] = sort(wxy, 'descend');
    sxx = xx;
    sxx(I(1:p)) = 0;
    [~,yxx] = max(W*sxx + 1-Y(:, idx));
    %fprintf('yy = %d, yxx = %d\n', yy, yxx);
    temp = zeros(K, 1);
    temp(yxx) = 1;
    W = (1-1/t)*W + (yxx~=yy)/lambda/t*(Y(:,idx) - temp)*sxx';
    
    nn = norm(W, 'fro');
    if nn > Fregion
        W = Fregion/nn*W;
    end
    if debug == 1 && mod(t, 1000) == 0
        [~, pred] = max(W*X, [], 1);
        train_err = mean(pred ~= trueY);
        test_err = 0;
        if exist('test_X', 'var') && exist('test_Y', 'var') && ~isempty(test_X) && ~isempty(test_Y)
            [~, pred] = max(W*test_X, [], 1);
            test_err = mean(pred~=trueTestY);
        end
        fprintf('Wnorm %f, train_err %f, test_err %f\n', nn, train_err, test_err);
    end
end

[~, pred] = max(W*X, [], 1);
train_err = mean(pred ~= trueY);

if exist('test_X', 'var') && exist('test_Y', 'var') && ~isempty(test_X) && ~isempty(test_Y)
    [~, pred] = max(W*test_X, [], 1);
    test_err = mean(pred~=trueTestY);
end