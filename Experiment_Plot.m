clear;
main_dir = 'helix_noisy';
standard_dir = 'sample/helix_seg_standard.nii';
standard_nii = load_nii( standard_dir );
standard_img = standard_nii.img;

dsc_all = zeros( 4, 10);
for noise_i = 1:10
    

    kmeans_dir = sprintf( '%s/noise_%d_kmeans.nii', main_dir, noise_i);
    kmeans_nii = load_nii( kmeans_dir );
    kmeans_img = kmeans_nii.img;
% 
    ncut_dir = sprintf( '%s/noise_%d_ncut.nii', main_dir, noise_i);
    ncut_nii = load_nii( ncut_dir );
    ncut_img = ncut_nii.img;
     
	ssc_dir = sprintf( '%s/noise_%d_ssc.nii', main_dir, noise_i);
    ssc_nii = load_nii( ssc_dir );
    ssc_img = ssc_nii.img;
    
    fpsc_dir = sprintf( '%s/noise_%d_fpsc.nii', main_dir, noise_i);
    fpsc_nii = load_nii( fpsc_dir );
    fpsc_img = fpsc_nii.img;
    

    dsc_kmeans = computeDSC( standard_img, kmeans_img);
    dsc_ncut = computeDSC( standard_img, ncut_img);
    dsc_ssc = computeDSC( standard_img, ssc_img);
    dsc_fpsc = computeDSC( standard_img, fpsc_img);
    
    dsc_all( :, noise_i) = [dsc_kmeans, dsc_ncut, dsc_ssc, dsc_fpsc ];

end

figure;
%noise_level = [40 35 30 25 20 15 10];
noise_level = 0:5:30;
noise_level = 0:5:30;

plot( noise_level, dsc_brain(1,:), 'go-');
hold on;
plot( noise_level, dsc_brain(2,:), 'y*-');
hold on;

plot( noise_level,  dsc_brain(3,:), 'c^-');
hold on;

plot( noise_level, dsc_brain(4,:), 'b*-');
hold off;

