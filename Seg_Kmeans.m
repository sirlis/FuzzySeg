function mat_seg = Seg_Kmeans( features, standard_img, label_ratio)

standard_img( standard_img == 0 ) = 2;
    
size_x = size( standard_img, 1 );
size_y = size( standard_img, 2 );
size_z = size( standard_img, 3 );
features_dim = size( features, 4 );
voxel_size = size_x * size_y * size_z;
    
standard_img_list = standard_img(:);
features_list = reshape( features, [voxel_size features_dim]);
    
[ s_features s_y ]= generateLabels( features_list, standard_img_list, label_ratio );  
addpath(genpath('RIM'));
addpath(genpath('minFunc_2012'));
p_params.max_class = 2; 
p_params.algo = 'linear'; 
p_params.lambda = 0.5;
cluster_result = RIM( features_list',s_features',s_y',p_params);
segment_cluster_y = cluster_result.alphas*( features_list');

voxel_size_brain = length( standard_img_list );
image_seg = zeros( 1, voxel_size_brain );
for j = 1:voxel_size_brain
    j_vector = reshape( segment_cluster_y(:,j), [1 2]) +reshape(cluster_result.bs, [1 2]);
    [ j_max j_cluster ] = max( j_vector );
    image_seg( j ) = j_cluster;
end

mat_seg = reshape( image_seg, [size_x size_y size_z] );
mat_seg(mat_seg==2) = 0;

end