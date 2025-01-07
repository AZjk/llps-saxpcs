for folder in cluster_results_part1 cluster_results_part2;
do
    for f in $(ls $folder);
    do
        # echo $f
        substring=$(echo "$folder" | cut -d'_' -f3)
        # echo $substring
        mv $folder/$f cluster_results/${substring}_$f
    done
done
