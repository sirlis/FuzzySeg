clear;
struct_dir = 'helix_clean/clean_tensorStruct.mat';
load( struct_dir );
tensorMat = tensorStruct.tensorMat;
features = tensorStruct.features;

helix_row = size( features, 1 );
helix_col = size( features, 2 );
helix_hei = size( features, 3 );
feature_dim = size( features, 4 );

helix_tensor = cell( helix_row, helix_col, helix_hei );
helix_tensor_vector = zeros( helix_row, helix_col, helix_hei, 6);

for i = 1:helix_row
    for j = 1:helix_col
        for k= 1:helix_hei
            
            ijk_features = squeeze( features( i, j, k, : ) );           
            ijk_FA = ijk_features( 5 );

            if ijk_FA<0.01
                ijk_tensor = [0.5 0 0 
                              0 0.6 0
                              0 0 0.5];
                ijk_tensor = [0.38 0 0 
                              0 0.6 0
                              0 0 0.38];
            else
                ijk_tensor = squeeze( tensorMat( i, j, k, :, : ) );
            end
            ijk_xx = ijk_tensor( 1, 1 );
            ijk_xy = ijk_tensor( 1, 2 );
            ijk_xz = ijk_tensor( 1, 3 );
            ijk_yy = ijk_tensor( 2, 2 );
            ijk_yz = ijk_tensor( 2, 3 );
            ijk_zz = ijk_tensor( 3, 3 );
            
            helix_tensor{ i, j, k } = ijk_tensor; 
            helix_tensor_vector( i, j, k, : ) = [ ijk_xx ijk_xy ijk_xz ijk_yy ijk_yz ijk_zz];

            [ijk_V,~] = eig( ijk_tensor );
            ijk_e = eig( ijk_tensor );
            ijk_e = abs( ijk_e );
            ijk_tensor = ijk_V*diag( ijk_e )*ijk_V';
            
            [~, e_max ]= max( ijk_e );
            ijk_e = sort( ijk_e, 'descend' );
            ijk_e1 = ijk_e( 1 );
            ijk_e2 = ijk_e( 2 );
            ijk_e3 = ijk_e( 3 );
            if ijk_e1>0
                
                ijk_ADC=( ijk_e1 + ijk_e2 + ijk_e3 )/3;
                ijk_FA=(sqrt(3*((ijk_e1-ijk_ADC)^2+(ijk_e2-ijk_ADC)^2+(ijk_e3-ijk_ADC)^2)))/(sqrt(2*(ijk_e1^2+ijk_e2^2+ijk_e3^2))); 
                
                if ijk_FA<1
                    
                    pd_v1 = ijk_V( 1, e_max );
                    pd_v2 = ijk_V( 2, e_max );
                    pd_v3 = ijk_V( 3, e_max ); 
                            
                    ijk_VR = ijk_e1*ijk_e2*ijk_e3/ijk_ADC;
                    ijk_RA = sqrt( (ijk_e1-ijk_ADC)^2 + (ijk_e2-ijk_ADC)^2 + (ijk_e3-ijk_ADC)^2 )/(sqrt(3)*ijk_ADC);
                    ijk_RD = ( ijk_e2 + ijk_e3 )/ 2;                %Radial Diffusivity
                    ijk_CL = ( ijk_e1 - ijk_e2) /(ijk_e1 + ijk_e2 + ijk_e3);        %Linearity Anisotropy
                    ijk_CP = 2*( ijk_e2 - ijk_e3) /(ijk_e1 + ijk_e2 + ijk_e3);        %Planarity Anisotropy
                    ijk_CS = 3*ijk_e3 /(ijk_e1 + ijk_e2 + ijk_e3);      %Spherical Anisotropy
                    ijk_AA = acosd( ijk_ADC*(ijk_e1 + ijk_e2 + ijk_e3) / sqrt(3 *ijk_ADC^2*(ijk_e1^2 + ijk_e2^2 + ijk_e3^2)))/180;%Angular Anisotropy
                    ijk_temp = ( log( ijk_e1) + log( ijk_e2) + log( ijk_e3) )/3;
                    ijk_AiA = sqrt( ( log( ijk_e1) - ijk_temp )^2+ ( log( ijk_e2) - ijk_temp )^2 + ( log( ijk_e3) - ijk_temp )^2 );%Aitchison Anisotropy(AitA)
                    ijk_LA = sqrt( ( log( ijk_e1) - log(ijk_ADC) )^2+ ( log( ijk_e2) - log(ijk_ADC)  )^2 + ( log( ijk_e3) - log(ijk_ADC)  )^2 );%Logarithmic Anisotropy(LA)
                    ijk_MA = sqrt( ( (sqrt(ijk_e1)-sqrt(ijk_ADC) )^2 +(sqrt(ijk_e2)-sqrt(ijk_ADC) )^2 +(sqrt(ijk_e3)-sqrt(ijk_ADC) )^2 ) / (ijk_e1 + ijk_e2 + ijk_e3)); %Matusita Anisotropy (MA)
                    ijk_KL = sqrt(3/2)*sqrt( log( ( (ijk_e1/ijk_ADC+ijk_e2/ijk_ADC+ijk_e3/ijk_ADC )/3)*((ijk_ADC/ijk_e1+ijk_ADC/ijk_e2+ijk_ADC/ijk_e3)/3) ) ); %Kullback¨CLeibler    Anisotropy(KLA)
                    ks_v1 = pd_v1^2 - pd_v2^2;
                    ks_v2 = 2* pd_v1 * pd_v2;
                    ks_v3 = 2* pd_v1 * pd_v3;
                    ks_v4 = 2* pd_v2 * pd_v3;
                    ks_v5 =  (2*pd_v3^2 - pd_v1^2 - pd_v2^2)/sqrt(3);
                    ijk_features = [ ijk_e1, ijk_e2, ijk_e3, ijk_ADC,ijk_FA, ijk_VR, ijk_RA, ijk_RD,ijk_CL,ijk_CP,ijk_CS,ijk_AA,ijk_AiA, ijk_LA,ijk_MA,ijk_KL,  ks_v1, ks_v2, ks_v3, ks_v4, ks_v5];
                    features( i,j,k,: )= ijk_features;
                    tensorMat(i,j,k,:,:) = ijk_tensor;
                end
            end
        end
    end
end

%display 
helix_tensor_slice = helix_tensor( 1:38, 1:38, 20 );
displayTensor( helix_tensor_slice )
% tensor_dir = sprintf( '%s/helix_tensor_cell.mat', main_dir );
% save( tensor_dir, 'helix_tensor');

main_dir = 'helix_test';

tensorStruct = struct;
tensorStruct.tensorMat = tensorMat;
tensorStruct.features = features;
struct_dir = sprintf( '%s/tensorStruct.mat', main_dir );
save( struct_dir, 'tensorStruct' );

%Convert tensor to inr
inr_dir =  sprintf('%s/helix_tensor.inr.gz', main_dir );
WriteInrTensorData( helix_tensor_vector, [ 1 1 1], inr_dir)

sample_dir = sprintf( '%s/dti_fa.nii', main_dir );
sample_nii = load_nii( sample_dir );

sample_nii.hdr.dime.dim = [ 3 helix_row helix_col helix_hei 1 1 1 1 ];
sample_nii.hdr.dime.pixdim = [ 0, 1, 1, 1, 1, 0, 0, 0 ];
sample_nii.hdr.hist.srow_x = [ 1, 0, 0, 0 ];
sample_nii.hdr.hist.srow_y = [ 0, 1, 0, 0 ];
sample_nii.hdr.hist.srow_z = [ 0, 0, 1, 0 ];
sample_nii.hdr.hist.originator = [ 0, 0, 0, 0 ];
sample_nii.img = squeeze( features( :, :, :, 5 ) );
sample_nii.fileprefix = sprintf( '%s/helix_FA.nii', main_dir );
save_nii( sample_nii, sample_nii.fileprefix );

