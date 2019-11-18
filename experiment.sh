rm -f *.pkl
# for DATASET in political_table transistor
# do
#     for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#     do
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#     done
# done

# for DATASET in cdr
# do
#     for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#     do
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#     done
# done

for DATASET in president_mix transistor
do
    for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
        make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
        make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
    done
done


for DATASET in president_mix transistor
do
    for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        make data=${DATASET} threshold=${T} gt_level=span snuba_exp
        make data=${DATASET} threshold=${T} gt_level=span snuba_exp
        make data=${DATASET} threshold=${T} gt_level=span snuba_exp
    done
done

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



# for DATASET in transistor
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


# for DATASET in transistor 
# do
#     for ESIZE in 1 3 5 7 9 11 13 15 17
#     do
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#     done
# done


# for DATASET in president_mix 
# do
#     for ESIZE in 1 3 5 7 9 11 13 15 17
#     do
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#         make example=${ESIZE} data=${DATASET} gt_level=span snuba_exp
#     done
# done