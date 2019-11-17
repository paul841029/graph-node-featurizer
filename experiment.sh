# for DATASET in political_table president_mix transistor
# do
#     for ESIZE in 1 3 5 10 15 20
#     do
#         make example=${ESIZE} data=${DATASET} gt_level=cell snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=cell snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=cell snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=cell snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=cell snuba_exp 
#     done
# done

# for DATASET in cdr
# do
#     for ESIZE in 1 3 5 10 15 20 30 40 60 70
#     do
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp 
#     done
# done

for DATASET in transistor president_mix
do
    for ESIZE in 1 3 5 7 9 11 13 15 17
    do
        make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
        make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
        make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
        make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
        make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
    done
done

