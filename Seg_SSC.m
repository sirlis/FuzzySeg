function mat_seg = Seg_SSC( features, standard_img, label_ratio, delta )
 
size_x = size( standard_img, 1 );
size_y = size( standard_img, 2 );
size_z = size( standard_img, 3 );
features_dim = size( features, 4 );
voxel_size = size_x * size_y * size_z;

standard_img_list = standard_img(:);
features_list = reshape( features, [voxel_size features_dim]);
standard_img_list( standard_img_list ==0 ) = -1;
  
[ s_features s_y s_index]= generateLabels( features_list, standard_img_list, label_ratio );
    
mat_normLap = constructGraph( features, delta );

label_list = zeros( 1, voxel_size );
label_list( s_index ) = s_y;
    
 mat_f = solveQuadprog(label_list,mat_normLap);
 mat_seg = reshape(mat_f, [size_x size_y size_z]);
mat_seg( mat_seg == - 1 ) = 0;

clc, clear all, close all

d = 1000;  
r0 = 0.6; % initial downsampling size to reduce computations
rp = 1; % downsampling size for downsampled features
D = 17; % projection deimension
SubDim = min([d 9]);

dix = find(M == d);
iter = 3; % number of classification trials
thr = 0.05; % threshold for noisy recovery via CVX
N = size(I,3); % number of images for each subject
n = size(I,4); % number of subjects/classes
NT = N - d; % Number of test images chosen from each subject

% scale for downsampling images 1/16==132, 1/8==504
Y = [];
B = ImageDownSample(I,r0);
Bd = ImageDownSample(I,rp);

for k = 1:iter
    Ytrain = [];
    Ytest = [];
    for i = 1:n
        Trn = Tr{dix}(:,i,k);
        Tst = Ts{dix}(:,i,k);
        Ytrain = [Ytrain B(:,Trn,i)];
        Ytest(:,:,i) = B(:,Tst,i);
    end
    Txt = strcat('Yale_','e',num2str(thr),'G',num2str(n),'D',num2str(D),'d',num2str(d));
    
    
    % run different methods using using projections with eigenfaces
    [Y,U,y_mean] = Eigenface(Ytrain,D);
    Y = MatrixNormalize(Y);
    for i = 1:n
        for j = 1:size(Tst,1)
            y = U' * (Ytest(:,j,i) - y_mean) ./ norm(U' * (Ytest(:,j,i) - y_mean) );
            
            sub = i;
            Tn = d * ones(1,n);
            [NN,err_NN] = NearestNeighbors(Y,y,Tn,sub);
            facerec{1}(i,j,k,1) = NN;
            [NS,err_NS] = NearestSubspace(Y,y,Tn,SubDim,sub);
            facerec{1}(i,j,k,2) = NS;
            q = 1;
            [Bsr1_L1,Bsr2_L1,err_Bsr1_L1,err_Bsr2_L1] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{1}(i,j,k,3) = Bsr1_L1;
            facerec{1}(i,j,k,4) = Bsr2_L1;
            q = 2;
            [Bsr1_L2,Bsr2_L2,err_Bsr1_L2,err_Bsr2_L2] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{1}(i,j,k,5) = Bsr1_L2;
            facerec{1}(i,j,k,6) = Bsr2_L2;

            eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])
        end
    end
    eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])
    
    
    % run different methods using projections with randomfaces
    [Y,U] = Randomface(Ytrain,D);
    Y = MatrixNormalize(Y);
    for i = 1:n
        for j = 1:size(Tst,1)
            y = U' * Ytest(:,j,i) ./ norm(U' * Ytest(:,j,i));
            
            sub = i;
            Tn = d * ones(1,n);
            [NN,err_NN] = NearestNeighbors(Y,y,Tn,sub);
            facerec{2}(i,j,k,1) = NN;
            [NS,err_NS] = NearestSubspace(Y,y,Tn,SubDim,sub);
            facerec{2}(i,j,k,2) = NS;
            q = 1;
            [Bsr1_L1,Bsr2_L1,err_Bsr1_L1,err_Bsr2_L1] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{2}(i,j,k,3) = Bsr1_L1;
            facerec{2}(i,j,k,4) = Bsr2_L1;
            q = 2;
            [Bsr1_L2,Bsr2_L2,err_Bsr1_L2,err_Bsr2_L2] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{2}(i,j,k,5) = Bsr1_L2;
            facerec{2}(i,j,k,6) = Bsr2_L2;

            eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])
        end
    end
    eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])
    
    
    % run different methods using downsampling
    Ytrain = [];
    Ytest = [];
    for i = 1:n
        Trn = Tr{dix}(:,i,k);
        Tst = Ts{dix}(:,i,k);
        Ytrain = [Ytrain Bd(:,Trn,i)];
        Ytest(:,:,i) = Bd(:,Tst,i);
    end
    Y = MatrixNormalize(Ytrain);
    
    for i = 1:n
        for j = 1:size(Tst,1)
            y = Ytest(:,j,i) ./ norm( Ytest(:,j,i) );
            
            sub = i;
            Tn = d * ones(1,n);
            [NN,err_NN] = NearestNeighbors(Y,y,Tn,sub);
            facerec{3}(i,j,k,1) = NN;
            [NS,err_NS] = NearestSubspace(Y,y,Tn,SubDim,sub);
            facerec{3}(i,j,k,2) = NS;
            q = 1;
            [Bsr1_L1,Bsr2_L1,err_Bsr1_L1,err_Bsr2_L1] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{3}(i,j,k,3) = Bsr1_L1;
            facerec{3}(i,j,k,4) = Bsr2_L1;
            q = 2;
            [Bsr1_L2,Bsr2_L2,err_Bsr1_L2,err_Bsr2_L2] = BlockSparse(Y,y,Tn,sub,thr,q);
            facerec{4}(i,j,k,5) = Bsr1_L2;
            facerec{4}(i,j,k,6) = Bsr2_L2;

            eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])
        end
    end
    eval(['save ' Txt '.mat facerec n D d NT r0 Tr Ts'])   
end


% compute the overal statistics of the results
for t = 1:3
    for p = 1:size(facerec{t},4)
        for k = 1:size(facerec{t},3)
            facerecTot(k,p,t) = sum( sum( facerec{t}(:,:,k,p) ) ) / ( size(facerec{t},1) * size(facerec{t},2) );
        end
        avgrecrate(p,t) = mean(facerecTot(:,p,t));
        maxrecrate(p,t) = max(facerecTot(:,p,t));
    end
end
eval(['save ' Txt '.mat facerec facerecTot avgrecrate maxrecrate n D d NT r0 Tr Ts'])




end