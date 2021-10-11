clear;
struct_dir = 'helix_clean/tensorStruct.mat';
load( struct_dir );
tensorMat = tensorStruct.tensorMat /1000;

helix_row = size( tensorMat, 1 );
helix_col = size( tensorMat, 2 );
helix_hei = size( tensorMat, 3 );

% eigVal = zeros( helix_row, helix_col, helix_hei, 3 );
eigVec = zeros( helix_row, helix_col, helix_hei ,3 );
FA = zeros( helix_row, helix_col, helix_hei );
% ADC = zeros( helix_row, helix_col, helix_hei );
% RA = zeros( helix_row, helix_col, helix_hei );
% VR = zeros( helix_row, helix_col, helix_hei );

for i = 1:helix_row
    for j = 1:helix_col
        for k= 1:helix_hei
            
            ijk_tensor = squeeze( tensorMat( i, j, k, :, : ) );
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
                    
                    ijk_VR = ijk_e1*ijk_e2*ijk_e3/ijk_ADC;
                    ijk_RA = sqrt( (ijk_e1-ijk_ADC)^2 + (ijk_e2-ijk_ADC)^2 + (ijk_e3-ijk_ADC)^2 )/(sqrt(3)*ijk_ADC);
%                     eigVal(i,j,k,:) = ijk_e;
                    FA(i,j,k) = ijk_FA;
%                     ADC(i,j,k) = ijk_ADC;
%                     VR(i,j,k) = ijk_VR;
%                     RA(i,j,k) = ijk_RA;
                    eigVec(i,j,k,1) = ijk_V( 1, e_max );
                    eigVec(i,j,k,2)= ijk_V( 2, e_max );
                    eigVec(i,j,k,3) = ijk_V( 3, e_max );  

                end
                
            end
        end
    end
end
 
sample_nii_dir = 'sample/dti_fa_color.nii';
sample_nii = load_nii( sample_nii_dir );
sample_img = double(sample_nii.img);

sample_nii.hdr.dime.dim = [ 3 helix_row helix_col helix_hei 1 1 1 1];
sample_nii.hdr.dime.pixdim= [0, 1, 1, 1, 1, 0, 0, 0];
sample_nii.hdr.hist.srow_x = [1, 0, 0, 0];
sample_nii.hdr.hist.srow_y = [0, 1, 0, 0];
sample_nii.hdr.hist.srow_y = [0, 0, 1, 0];
sample_nii.hdr.hist.originator = [0 0 0 0 0];


FA_Color = zeros( helix_row, helix_col, helix_hei, 3 );
eigVec = abs(eigVec);
i_min = min( eigVec(:) );
i_max = max( eigVec(:) );
i_gap = i_max - i_min;

for i = 1:3
    
    i_eigVec = squeeze( eigVec( :, :, :, i ) );
    FA_Color( :, :, :, i ) = (i_eigVec-i_min)*255/i_gap;

end

helix_nii_dir = 'helix_clean/helix_FA_Color.nii';
sample_nii.img = FA_Color;
sample_nii.fileprefix = helix_nii_dir;
save_nii( sample_nii, sample_nii.fileprefix);


r = squeeze( abs(eigVec( :, :, 20, 1 )) );
g = squeeze( abs(eigVec( :, :, 20, 2 )) );
b = squeeze( abs(eigVec( :, :, 20, 3 )) );
FA = squeeze( FA( :, :, 20) );
rgb=cat(3,r,g,b);
FAmap3=cat(3,FA,FA,FA); %needed to combine with the colormap, need 3D.
cm=rgb.*FAmap3; %FA weighting for colormap
figure, imshow (cm, [min(cm(:)) max(cm(:))]); 
title('FA Color Map');

figure, imshow (FA, [0 1]); 
title('FA Map');