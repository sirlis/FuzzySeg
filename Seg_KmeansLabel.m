function mat_seg = Seg_KmeansLabel( features )
    
size_x = size( features, 1 );
size_y = size( features, 2 );
size_z = size( features, 3 );
features_dim = size( features, 4 );
voxel_size = size_x * size_y * size_z;
    
features_list = reshape( features, [voxel_size features_dim]);

label_dir = 'sample/helix_label.nii';
label_nii = load_nii( label_dir );
label_img = label_nii.img;
label_img_list = label_img(:);
label_img_list( label_img_list ==3) = 2;

label_index = find( label_img_list>0 );
s_features = features_list( label_index, : );
s_y = label_img_list( label_index);

% [ s_features s_y ]= generateLabels( features_list, standard_img_list, label_ratio ); 


addpath(genpath('RIM'));
addpath(genpath('minFunc_2012'));
p_params.max_class = 2; 
p_params.algo = 'linear'; 
p_params.lambda = 0.5;
cluster_result = RIM( features_list',s_features',s_y',p_params);
segment_cluster_y = cluster_result.alphas*( features_list');

image_seg = zeros( 1, voxel_size );
for j = 1:voxel_size
    j_vector = reshape( segment_cluster_y(:,j), [1 2]) +reshape(cluster_result.bs, [1 2]);
    [ j_max j_cluster ] = max( j_vector );
    image_seg( j ) = j_cluster;
end

mat_seg = reshape( image_seg, [size_x size_y size_z] );
mat_seg(mat_seg==2) = 0;

end