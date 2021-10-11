function seg_dsc = computeDSC( seg_manual, seg_auto)

    seg_overlap= seg_manual.*seg_auto;

    sum_overlap = sum(seg_overlap(:));
    sum_manual = sum(seg_manual(:));
    sum_auto = sum(seg_auto(:));

    seg_dsc = 2*sum_overlap/(sum_manual+sum_auto);
end