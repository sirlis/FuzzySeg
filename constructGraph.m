function mat_normLap = constructGraph( features,delta )

ADC = features( :, :, :, 4 );

size_x = size(ADC,1);
size_y = size(ADC,2);
size_z = size(ADC,3);
size_xyz = size_x * size_y * size_z;
size_xy = size_x * size_y;

mat_W = sparse(size_xyz,size_xyz);

neighour_conn=[ 1  0 0; 0 1 0; 0 0 1];
neighour_num = size(neighour_conn,1);

for i = 1:size_x
    for j = 1:size_y
        for k =1:size_z
            
            m_features = squeeze(features(i,j,k,:));
            if ADC(i,j,k)==0
                continue;
            end
            
            m_neighour= repmat([i j k], [neighour_num 1]) + neighour_conn;    
            m_index = (k-1)*size_xy + (j-1)*size_x + i;
            for s=1:neighour_num
                n_i = m_neighour(s,1);
                n_j = m_neighour(s,2);
                n_k = m_neighour(s,3);
                if n_i<=size_x && n_j<=size_y && n_k<=size_z

                    n_index = (n_k-1)*size_xy + (n_j-1)*size_x + n_i;
                    if ADC(n_i,n_j,n_k)==0
                            continue;
                    end
                    n_features = squeeze(features(n_i,n_j,n_k,:));
                    mn_distance = (m_features- n_features)'*(m_features- n_features);
                    weight_mn = exp(-mn_distance*delta);
                    mat_W(m_index,n_index) = weight_mn;
                    mat_W(n_index,m_index) = weight_mn;
                end  
            end
            
            
        end
    end
end

mat_D = sparse(size_xyz,size_xyz);
mat_msD = sparse(size_xyz,size_xyz);

for i = 1:size_xyz 
	weight_sum = sum(mat_W(i,:));
	mat_D(i,i) = weight_sum;
	if weight_sum~=0
       	mat_msD(i,i)  = weight_sum^(-1/2);
    end
end

mat_Ind = speye(size_xyz);
mat_nL_tril= tril(mat_Ind - mat_msD*mat_W*mat_msD);
mat_normLap = mat_nL_tril+ tril(mat_nL_tril,-1)';

end