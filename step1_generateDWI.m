clear;
struct_dir = 'helix_s1/tensorStruct.mat';
load( struct_dir );
tensorMat = tensorStruct.tensorMat / 1000;

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

DWI_dir = 'helix_noisy/dwi_clean.mat';
save( DWI_dir, 'helix_dwi');

