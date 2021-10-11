function mat_seg = Seg_GraphCutLabel( features, delta )
 
size_x = size( features, 1 );
size_y = size( features, 2 );
size_z = size( features, 3 );
  
label_dir = 'sample/helix_label.nii';
label_nii = load_nii( label_dir );
label_img = label_nii.img;
label_img_list = label_img(:);
label_img_list( label_img_list == 3 ) = -1;

mat_normLap = constructGraph( features, delta );

    
mat_f = solveQuadprog(label_img_list,mat_normLap);
mat_seg = reshape(mat_f, [size_x size_y size_z]);
mat_seg( mat_seg == - 1 ) = 0;

end