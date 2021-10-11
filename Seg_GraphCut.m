function mat_seg = Seg_GraphCut( features, standard_img, label_ratio, delta )
 
size_x = size( standard_img, 1 );
size_y = size( standard_img, 2 );
size_z = size( standard_img, 3 );
features_dim = size( features, 4 );
voxel_size = size_x * size_y * size_z;

standard_img_list = standard_img(:);
features_list = reshape( features, [voxel_size features_dim]);
standard_img_list( standard_img_list ==0 ) = -1;
  
[ s_features s_y s_index]= generateLabels( features_list, standard_img_list, label_ratio );
    
% mat_normLap = constructGraph( features, delta );
mat_normLap = constructGraphAll( features, delta );

label_list = zeros( 1, voxel_size );
label_list( s_index ) = s_y;
    
 mat_f = solveQuadprog(label_list,mat_normLap);
 mat_seg = reshape(mat_f, [size_x size_y size_z]);
mat_seg( mat_seg == - 1 ) = 0;

end