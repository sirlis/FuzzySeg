clear;
struct_dir = 'helix_clean/tensorStruct.mat';
load( struct_dir );
tensorMat = tensorStruct.tensorMat /1000;

helix_row = size( tensorMat, 1 );
helix_col = size( tensorMat, 2 );
helix_hei = size( tensorMat, 3 );

bval = 1000;
helix_gradient = [
0,         0,         0
1.00000,   0.41425,   -0.41425
1.00000,   -0.41425,   -0.41425
1.00000,   -0.41425,   0.41425
1.00000,   0.41425,   0.41425
0.41425,   0.41425,   1.00000
0.41425,   1.00000,   0.41425
0.41425,   1.00000,   -0.41425
0.41425,   0.41425,   -1.00000
0.41425,   -0.41425,   -1.0000
0.41425,   -1.00000,   -0.41425
0.41425,   -1.00000,   0.41425
0.41425,   -0.41425,   1.00000
];
gradient_number = size( helix_gradient, 1 );
helix_dwi = zeros( helix_row, helix_col, helix_hei,gradient_number );
helix_dwi( : , :, :, 1 ) = 100;
S0 = squeeze( helix_dwi( : , :, :, 1 ) );
for i = 1:helix_row
    for j = 1:helix_col
        for k= 1:helix_hei
            
            ijk_tensor = squeeze( tensorMat( i, j, k, :, : ) );
            ijk_S0 = S0( i, j, k, 1 );
            for g = 2:gradient_number

                gradient_g = reshape( squeeze( helix_gradient( g, : ) ), [ 1 3] );
                ijk_Si = ijk_S0 * exp( -bval*gradient_g * ijk_tensor*gradient_g');
                helix_dwi( i, j, k, g ) = ijk_Si;
            end
            
        end
    end
end

DTIdata=struct();
for i=1:gradient_number
    DTIdata(i).VoxelData = squeeze( helix_dwi( :, :, :, i) ); 
    DTIdata(i).Gradient = helix_gradient(i,:);
    DTIdata(i).Bvalue = bval;
end

% Constants DTI
parametersDTI=[];
parametersDTI.BackgroundTreshold=0;
parametersDTI.WhiteMatterExtractionThreshold=0.10;
parametersDTI.textdisplay=true;

% Perform DTI calculation
[ADC,FA,EigVec,helix_tensor]=computeTensor(DTIdata,parametersDTI);

% helix_tensor = cell( helix_row, helix_col, helix_hei );
helix_tensor_vector = zeros( helix_row, helix_col, helix_hei, 6 );
tensorNoisyMat = zeros( helix_row, helix_col, helix_hei, 3, 3 );
feature_dim = 24;

features = zeros( helix_row, helix_col, helix_hei,feature_dim );

main_dir = 'helix_noisy';
for i = 1:helix_row
    for j = 1:helix_col
        for k= 1:helix_hei
            
            ijk_tensor = helix_tensor{ i, j, k };
            ijk_xx = ijk_tensor( 1, 1 );
            ijk_xy = ijk_tensor( 1, 2 );
            ijk_xz = ijk_tensor( 1, 3 );
            ijk_yy = ijk_tensor( 2, 2 );
            ijk_yz = ijk_tensor( 2, 3 );
            ijk_zz = ijk_tensor( 3, 3 );
            
            helix_tensor_vector( i, j, k, : ) = [ ijk_xx ijk_xy ijk_xz ijk_yy ijk_yz ijk_zz];
%             helix_tensor{ i, j, k } = ijk_tensor; 
            
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
                    ijk_KL = sqrt(3/2)*sqrt( log( ( (ijk_e1/ijk_ADC+ijk_e2/ijk_ADC+ijk_e3/ijk_ADC )/3)*((ijk_ADC/ijk_e1+ijk_ADC/ijk_e2+ijk_ADC/ijk_e3)/3) ) ); %Kullback?Leibler    Anisotropy(KLA)
                    ks_v1 = pd_v1^2 - pd_v2^2;
                    ks_v2 = 2* pd_v1 * pd_v2;
                    ks_v3 = 2* pd_v1 * pd_v3;
                    ks_v4 = 2* pd_v2 * pd_v3;
                    ks_v5 =  (2*pd_v3^2 - pd_v1^2 - pd_v2^2)/sqrt(3);
                    ijk_features = [ ijk_e1, ijk_e2, ijk_e3, ijk_ADC,ijk_FA, ijk_VR, ijk_RA, ijk_RD,ijk_CL,ijk_CP,ijk_CS,ijk_AA,ijk_AiA, ijk_LA,ijk_MA,ijk_KL,  ks_v1, ks_v2, ks_v3, ks_v4, ks_v5, i/helix_row, j/helix_col, k/helix_hei];
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
% 
tensorStruct = struct;
tensorStruct.tensorMat = tensorMat;
tensorStruct.features = features;
struct_dir = sprintf( '%s/test_tensorStruct.mat', main_dir );
save( struct_dir, 'tensorStruct' );

%Convert tensor to inr
inr_dir =  sprintf('%s/helix_tensor.inr.gz', main_dir );
WriteInrTensorData( helix_tensor_vector*1000, [ 1 1 1], inr_dir)

sample_dir =  'sample/dti_fa.nii';
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

sample_nii.img = squeeze( features( :, :, :, 5 ) > 0.01 );
sample_nii.fileprefix = sprintf( '%s/helix_seg_standard.nii', main_dir );
save_nii( sample_nii, sample_nii.fileprefix );

sample_nii.img = squeeze( features( :, :, :, 4 ) );
sample_nii.fileprefix = sprintf( '%s/helix_MD.nii', main_dir );
save_nii( sample_nii, sample_nii.fileprefix );
