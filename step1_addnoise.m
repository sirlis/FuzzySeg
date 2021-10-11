clear;

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

DWI_dir = 'helix_noisy/dwi_clean.mat';
load( DWI_dir );
helix_row = size( helix_dwi, 1 );
helix_col = size( helix_dwi, 2 );
helix_hei = size( helix_dwi, 3 );
gradient_number = size( helix_dwi, 4 );

helix_dwi_noisy = zeros( helix_row, helix_col, helix_hei,gradient_number );
 
noise_level = 2:2:20;
for noise_i = 1:10
    
    noise_level_i = noise_level( noise_i );
    %generate dwi
    for gradient_i = 1:gradient_number
        
        Si = helix_dwi( :, :, :, gradient_i );
        Si_r = ricernd( Si, noise_level_i ); 
        helix_dwi_noisy( :, :, :, gradient_i ) = Si_r;
    end
    
    %estimate dti
    DTIdata=struct();
    for i=1:gradient_number
        DTIdata(i).VoxelData = squeeze( helix_dwi_noisy( :, :, :, i) ); 
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
    
    helix_tensor_vector = zeros( helix_row, helix_col, helix_hei, 6 );
    tensorMat = zeros( helix_row, helix_col, helix_hei, 3, 3 );
    feature_dim = 21;

    features = zeros( helix_row, helix_col, helix_hei,feature_dim );

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
    
	noise_i_dwi_dir = sprintf( 'helix_noisy/noise_%d_dwi.mat', noise_i);
    noise_i_dti_dir = sprintf( 'helix_noisy/noise_%d_dti.mat', noise_i);
    noise_i_struct_dir = sprintf( 'helix_noisy/noise_%d_struct.mat', noise_i);
    noise_i_inr_dir = sprintf( 'helix_noisy/noise_%d_inr.inr.gz', noise_i);
    
    save( noise_i_dwi_dir, 'helix_dwi_noisy' );
    save( noise_i_dti_dir, 'helix_tensor' );
    tensorStruct = struct;
    tensorStruct.tensorMat = tensorMat;
    tensorStruct.features = features;
    save( noise_i_struct_dir, 'tensorStruct' );

    %Convert tensor to inr
    WriteInrTensorData( helix_tensor_vector*1000, [ 1 1 1], noise_i_inr_dir);
    
end
 
