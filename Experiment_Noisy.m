clear;
main_dir = 'helix_noisy';
label_ratio = 0.2;
standard_dir = 'sample/helix_seg_standard.nii';
standard_nii = load_nii( standard_dir );
standard_img = standard_nii.img(:, :, :);

for noise_i = 1:10
    
     noise_i_struct_dir = sprintf( '%s/noise_%d_struct.mat', main_dir, noise_i);
     load( noise_i_struct_dir );
     tensorMat = tensorStruct.tensorMat(:, :, :, :);
    features = tensorStruct.features(:, :, :, :);

    seg_kmeans = Seg_Kmeans( features, standard_img, label_ratio );
    seg_dsc_kmeans = computeDSC( standard_img, seg_kmeans)
%     kmeans_dir = sprintf( '%s/noise_%d_kmeans.nii', main_dir, noise_i);
%     standard_nii.img =  seg_kmeans;
%     standard_nii.fileprefix = kmeans_dir;
%     save_nii( standard_nii, standard_nii.fileprefix); 
    
%     delta = 10;
%     %label_ratio = 0.02+0.004*(10-noise_i);
% 
%     seg_ncut = Seg_GraphCut( features, standard_img, label_ratio, delta );
% %     seg_ncut =Seg_GraphCutLabel( features, delta );
%     seg_dsc_ncut = computeDSC( standard_img, seg_ncut)
%  
%     ncut_dir = sprintf( '%s/noise_%d_ncut.nii', main_dir, noise_i);
%     standard_nii.img =  seg_ncut ;
%     standard_nii.fileprefix = ncut_dir;
%    save_nii( standard_nii, standard_nii.fileprefix); 
    
%    label_ratio = 0.03; 
%     delta = 0.05;
%     seg_ssc = Seg_GraphCut( features, standard_img, label_ratio, delta );
%     seg_dsc_ssc = computeDSC( standard_img, seg_ssc)
% 
%     ssc_dir = sprintf( '%s/noise_%d_ssc.nii', main_dir, noise_i);
%     standard_nii.img =  seg_ssc ;
%     standard_nii.fileprefix = ssc_dir;
%     save_nii( standard_nii, standard_nii.fileprefix); 
%     
%      delta = 1;
%      seg_fpsc = Seg_GraphCut( features, standard_img, label_ratio, delta );
% 	fpsc_dir = sprintf( '%s/noise_%d_fpsc.nii', main_dir, noise_i);
%     fpsc_nii = load_nii( fpsc_dir );
%    seg_tsc = fpsc_nii.img;
%     seg_dsc_fpsc = computeDSC( standard_img, seg_tsc)
%     
%     standard_nii.img =  seg_fpsc ;
%      standard_nii.fileprefix = fpsc_dir;
%      save_nii( standard_nii, standard_nii.fileprefix); 
end

 