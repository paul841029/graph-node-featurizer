rm -f *.pkl

# make data=political_table example=1 threshold=0 gt_level=cell snuba_exp
for DATASET in political_table president_mix transistor
do
    for ne in 1 3 5 7 9 11
    do
        make data=${DATASET} example=${ne} gt_level=cell ratio=1 snuba_exp
        make data=${DATASET} example=${ne} gt_level=cell ratio=1 snuba_exp
        make data=${DATASET} example=${ne} gt_level=cell ratio=1 snuba_exp
    done
done


for DATASET in political-text-pob
do
    for ne in -1 1 3 5 7 9 11
    do
        make data=${DATASET} example=${ne} gt_level=span ratio=10 snuba_exp
        make data=${DATASET} example=${ne} gt_level=span ratio=10 snuba_exp
        make data=${DATASET} example=${ne} gt_level=span ratio=10 snuba_exp
    done
done

for DATASET in political-text-pob
do
    for ne in 1 3 5 7 9 11
    do
        make data=${DATASET} example=${ne} gt_level=span ratio=1 snuba_exp
        make data=${DATASET} example=${ne} gt_level=span ratio=1 snuba_exp
        make data=${DATASET} example=${ne} gt_level=span ratio=1 snuba_exp
    done
done




# for DATASET in cdr
# do
#     for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#     do
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#     done
# done

# for DATASET in president_mix transistor
# do
#     for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#     do
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=cell snuba_exp
#     done
# done


# for DATASET in transistor president_mix
# do
#     for T in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#     do
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
#         make data=${DATASET} threshold=${T} gt_level=span snuba_exp
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
