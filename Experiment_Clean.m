clear;
main_dir = 'helix_clean';
label_ratio = 0.1; 

struct_dir = sprintf( '%s/tensorStruct.mat', main_dir );
load( struct_dir );
tensorMat = tensorStruct.tensorMat;
features = tensorStruct.features;
    
standard_dir = [main_dir,'/helix_seg_standard.nii'];
standard_nii = load_nii( standard_dir );
standard_img = standard_nii.img;

% seg_kmeans = Seg_Kmeans( features, standard_img, label_ratio );
seg_kmeans = Seg_KmeansLabel( features );

seg_dsc_kmeans = computeDSC( standard_img, seg_kmeans)
% 
% delta = 0.05;
% seg_ncut = Seg_GraphCut( features, standard_img, label_ratio, delta );
% 
% seg_dsc_ncut = computeDSC( standard_img, seg_ncut)
%      
% delta = 0.1;
% seg_ssc = Seg_GraphCut( features, standard_img, label_ratio, delta );
% seg_dsc_ssc = computeDSC( standard_img, seg_ssc)
%         


% kmeans_dir = sprintf( '%s/result_kmeans.nii',main_dir );
% standard_nii.img =  seg_kmeans ;
% standard_nii.fileprefix = kmeans_dir;
% save_nii( standard_nii, standard_nii.fileprefix); 
% 
% ncut_dir = sprintf( '%s/result_ncut.nii',main_dir );
% standard_nii.img =  seg_ncut ;
% standard_nii.fileprefix = ncut_dir;
% save_nii( standard_nii, standard_nii.fileprefix); 
% 
% ssc_dir = sprintf( '%s/result_ssc.nii',main_dir );
% standard_nii.img =  seg_kmeans ;
% standard_nii.fileprefix = ssc_dir;
% save_nii( standard_nii, standard_nii.fileprefix); 

delta = 1;
seg_fpsc = Seg_GraphCutLabel( features, delta );
seg_dsc_tsc = computeDSC( standard_img, seg_fpsc )

fpsc_dir = sprintf( '%s/result_fpsc.nii',main_dir );
standard_nii.img =  seg_fpsc ;
standard_nii.fileprefix = fpsc_dir;
save_nii( standard_nii, standard_nii.fileprefix); 
