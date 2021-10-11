function [model,final_loss] = RIM(X_unlabeled,X_labeled,y,params,init_model)
    D = size(X_unlabeled,1);
    
    [params,options] = process_options(params);
    
    % set initial conditions
    if nargin == 5
        result_vec = [init_model.alphas(:); init_model.bs];
    else
        alphas = randn(params.max_class,D)/params.alpha_divisor;
        bs = ones(params.max_class,1);%randn(params.max_class,1);
        result_vec = [alphas(:); bs];
    end
    
    if params.unsup_init % initialize using another clustering algorithm
        result_vec = init_unsup(result_vec, X_unlabeled, options, params);
    end
    
    % initialize the model using supervised objective on LABELED data
    if params.sup_init
        result_vec = init_sup(result_vec, X_unlabeled, X_labeled, y, params, options);
    end
    
    [result_vec,final_loss] = minFunc(@rim_cost,result_vec,options,X_unlabeled,X_labeled,y,params);
    
    model.alphas = reshape(result_vec(1:D*params.max_class),[params.max_class D]);
    model.bs = result_vec(D*params.max_class+1:end);
    
end

function result_vec = init_sup(result_vec, X_unlabeled, X_labeled, y, params, options)
    params.tau = 0;
    result_vec = minFunc(@rim_cost,result_vec,options,X_unlabeled,X_labeled,y,params);
end

function result_vec = init_unsup(result_vec, X_unlabeled, options, params)
        if isequal(params.algo,'linear') % use k-means
            unsup_labels = kmeans2(X_unlabeled',params.max_class)';
        elseif isequal(params.algo,'kernel') % use spectral clustering
            clusters = gcut(K,params.max_class);
            unsup_labels = zeros(1,size(X_unlabeled,2));
            for c = 1:length(clusters)
                unsup_labels(clusters{c}) = c;
            end
        end
        
        % train supervised classifier on initial cluster labels
        options.MAXITER = 50; % optimize loosely (few iterations)
        params.display_terms = false;
        params.tau = 0;
        result_vec = minFunc(@rim_cost,result_vec,options,[],X_unlabeled,unsup_labels,params);
end

function [params,options] = process_options(params)
    
    if ~isfield(params,'alpha_divisor')
        params.alpha_divisor = 1;
    end
    if ~isfield(params,'MAXITER')
        options.MAXITER = 1000;
    else
        options.MAXITER = params.MAXITER;
    end
    if ~isfield(params,'MAXFUNEVALS')
        options.MAXFUNEVALS = 2000;
    else
        options.MAXFUNEVALS = params.MAXFUNEVALS;
    end
    if ~isfield(params,'normalize_info')
        params.normalize_info = false;
    end
    if ~isfield(params,'display_terms')
        params.display_terms = false;
    end

    if isfield(params,'Method')
        options.Method = params.Method;
    else
        options.Method = 'lbfgs';
    end
    if isfield(params,'LS')
        options.LS = params.LS;
    else
        options.LS = 4;
    end
    if isfield(params,'USEMEX')
        options.USEMEX = params.USEMEX;
    else
        options.USEMEX = 0;
    end
    if ~isfield(params,'tau')
        params.tau = 1;
    end
    if ~isfield(params,'cost')
        params.cost = 'full';
    end
    if ~isfield(params,'sup_init')
        params.sup_init = false;
    end
    if ~isfield(params,'unsup_init')
        params.unsup_init = false;
    end
    
    options.Display = 'full'; %'excessive';
end
   