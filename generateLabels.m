function [ s_features s_y s_index]= generateLabels( features, standard_img, label_ratio )

i_seg =zeros( 1, 2);
i_seg(1) = min( standard_img(:) );
i_seg(2) = max( standard_img(:) );

s_index = [];
for i = 1:2
    
    i_seg_index = find( standard_img == i_seg(i) );
    i_seg_number = length( i_seg_index );
    i_select_number = round( i_seg_number *label_ratio);
    i_rand = randi( i_seg_number, 1, i_select_number );
    %i_seg_index = i_seg_index( i_rand+1 );
    i_seg_index = i_seg_index( i_rand );
    s_index = [s_index i_seg_index'];
end

s_features = features(s_index, :);
s_y = standard_img(s_index);
end