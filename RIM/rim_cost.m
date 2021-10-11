% X_unlabeled is the unlabeled data/kernel matrix
% y is the set of labels of the first length(y) data points. use y = [] for
%   purely unsupervised learning
% params holds maximum number of classes and regularization parameters

function [loss_value, the_grad] = rim_cost(current_vector, X_unlabeled, X_labeled, y, params)
    if params.tau > 0
        D = size(X_unlabeled,1);
    else
        D = size(X_labeled,1);
    end
    
    N_labeled = size(X_labeled,2);
    N_unlabeled = size(X_unlabeled,2);
    
    if N_labeled ~= length(y)
        error('Number of labels does not equal number of labeled examples');
    end
    
    n_class = params.max_class;
    
    alphas = reshape(current_vector(1:D*params.max_class),[params.max_class D]);
    bs = current_vector(D*params.max_class+1:end);
    
    if N_unlabeled > 0
        A_unlabeled = alphas*X_unlabeled;
    else
        A_unlabeled = zeros(n_class,0);
    end
    if N_labeled > 0
        A_labeled = alphas*X_labeled;
    else
        A_labeled = zeros(n_class,0);
    end
    
    if isequal(params.algo,'linear')
        dL_dalpha = 2*params.lambda*alphas;
        w_energy = params.lambda*sum(sum(alphas.^2));
    elseif isequal(params.algo,'kernel')
        % by convention labeled examples are ordered first
        dL_dalpha = 2*params.lambda*[A_labeled A_unlabeled];
        w_energy = params.lambda*trace([A_labeled A_unlabeled]*alphas');
    end
    dL_db = 0;
    
    A_unlabeled = A_unlabeled + bs(:,ones(1,N_unlabeled));
    A_max = max(A_unlabeled,[],1);
    A_unlabeled = A_unlabeled - A_max(ones(n_class,1),:);
    p_unlabeled = exp(A_unlabeled);
    Z_unlabeled = sum(p_unlabeled,1);
    p_unlabeled = p_unlabeled./Z_unlabeled(ones(n_class,1),:);  
    
    clear A_unlabeled;
    
    A_labeled = A_labeled + bs(:,ones(1,N_labeled));
    A_max = max(A_labeled,[],1);
    A_labeled = A_labeled - A_max(ones(n_class,1),:);
    p_labeled = exp(A_labeled);
    Z_labeled = sum(p_labeled,1);
    p_labeled = p_labeled./Z_labeled(ones(n_class,1),:);
    
    likelihood = 0;
    % if there are labeled values
    if N_labeled > 0
        sup_idx = sub2ind(size(A_labeled),y,1:N_labeled);
        likelihood = sum(A_labeled(sup_idx) - log(Z_labeled));
        
        label_matrix = zeros(n_class,N_labeled);
        label_idx = sub2ind(size(label_matrix),y,1:N_labeled);
        label_matrix(label_idx) = 1;
        
        dL_dalpha = dL_dalpha - (label_matrix - p_labeled)*(X_labeled');
        
        dL_db = -sum(label_matrix - p_labeled,2);
        
    end
    
    clear A_labeled;
    
    % if we are training on supervised set only (for initialization purposes, 
    %    since the purely supervised problem is convex)
    if (N_unlabeled == 0)||(params.tau==0) 
        loss_value = w_energy - likelihood;
    else
        if isequal(params.cost,'full')
           
            %nz_idx = p>0;
            zero_idx = (p_unlabeled==0);
            P = sum(p_unlabeled,2)/N_unlabeled;
            
            if isfield(params,'ref_dist')
                [foo, sort_idx] = sort(P,1,'descend');
                sorted_ref_dist(sort_idx) = params.ref_dist;
                P_norm = P./sorted_ref_dist;
            else
                P_norm = P;
            end
            
            %p_log_p_over_P = zeros(params.max_class,N_unlabeled,'single');
            rep_P = P_norm(:,ones(1,N_unlabeled));
            
            %p_log_p_over_P(nz_idx) = p(nz_idx).*log(p(nz_idx)./rep_P(nz_idx));
            p_log_p_over_P = p_unlabeled.*log(p_unlabeled./rep_P);
            p_log_p_over_P(zero_idx) = 0;
            p_log_p_over_P(find(P_norm==0),:) = 0;
            if params.display_terms
                CB = sum(P(P>0).*log(P(P>0)));
                PW = -sum(sum(p_unlabeled(p_unlabeled>0).*log(p_unlabeled(p_unlabeled>0))));
            end
            
            clear rep_P;
            
            KL = sum(p_log_p_over_P,1);
            
            rep_KL = KL(ones(n_class,1),:);

            matrix_term = p_log_p_over_P - p_unlabeled.*rep_KL;
            clear rep_KL;
            clear p_log_p_over_P;
             
            %dL_dalpha = dL_dalpha - params.tau*matrix_term*(X(:,N_labeled+1:end)');
            if params.normalize_info 
                normalizer = N_unlabeled;
            else
                normalizer = 1;
            end
            dL_dalpha = dL_dalpha - params.tau*(X_unlabeled*matrix_term')'/normalizer; % avoid transposing data matrix

            dL_db = dL_db - params.tau*sum(matrix_term,2)/normalizer;
            
            info = -params.tau*sum(KL)/normalizer;
            loss_value = w_energy - likelihood + info;
        elseif isequal(params.cost,'conditional')
            cond_ent = sum(p_unlabeled.*log(p_unlabeled),1);
            term_matrix = log(p_unlabeled) - cond_ent(ones(params.max_class,1),:);
            
            %dL_dalpha = dL_dalpha - params.tau*(p_unlabeled.*term_matrix)*(X_unlabeled');
            dL_dalpha = dL_dalpha - params.tau*(X_unlabeled*(p_unlabeled.*term_matrix)')';
            
            dL_db = dL_db - params.tau*sum(p_unlabeled.*term_matrix,2);
            
            loss_value = w_energy - likelihood - params.tau*sum(cond_ent);
        end
    end
    
    if params.display_terms
        display([' H_class: ' num2str(-CB) ' H_cond: ' num2str(PW/N_unlabeled) ' Reg: ' num2str(w_energy/N_unlabeled)]);
        figure(2);
        subplot(3,1,1);
        [P_sort,sort_idx] = sort(P,1,'descend');
        bar(P_sort);
        subplot(3,1,2);
        
        bar(diag(alphas(sort_idx,:)*alphas(sort_idx,:)'));
        subplot(3,1,3);
        bar(bs(sort_idx));
        
        [foo,y] = max(p_unlabeled,[],1);
        
        title(['K_{unlabeled}=' num2str(length(unique(y)))]);
    end
    
    the_grad = [dL_dalpha(:); dL_db];
    
end