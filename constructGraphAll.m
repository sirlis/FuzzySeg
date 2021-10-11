function mat_normLap = constructGraphAll( features,delta )

size_x = size( features, 1 );
size_y = size( features, 2 );
size_z = size( features, 3 );
feature_dim = size( features, 4 );
voxel_size = size_x * size_y * size_z;
   
features_list = reshape( features, [voxel_size feature_dim])';


aa=sum(features_list.*features_list); 
ab=features_list'*features_list; 
features_distance = repmat(aa',[1 size(aa,2)]) + repmat(aa,[size(aa,2) 1]) - 2*ab;

mat_W = exp(-features_distance*delta);


mat_D = zeros(voxel_size,voxel_size);
mat_msD = zeros(voxel_size,voxel_size);

for i = 1:voxel_size 
	weight_sum = sum(mat_W(i,:));
	mat_D(i,i) = weight_sum;
	if weight_sum~=0
       	mat_msD(i,i)  = weight_sum^(-1/2);
    end
end

clear mat_D;
mat_Ind = eye(voxel_size);
mat_nL_tril= tril(mat_Ind - mat_msD*mat_W*mat_msD);
clear mat_W;
clear mat_msD;
clear mat_Ind;
mat_normLap = mat_nL_tril+ tril(mat_nL_tril,-1)';
clear mat_nL_tril;

end