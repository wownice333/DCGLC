## Dual Contrastive Graph-Level Clustering with Multiple Cluster Perspectives Alignment

This is the official implementation of Dual Contrastive Graph-Level Clustering with Multiple Cluster Perspectives Alignment, IJCAI 2024.

### Dependencies

- python 3.8, pytorch, torch-geometric, torch-sparse, numpy, scikit-learn

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Reproduce graph data results

To generate results

    python demo_DCGLC.py --DS BZR --eval True

To train DCGLC without loading saved weight files

    python demo_DCGLC.py --DS BZR --eval False


If you've found DCGLC useful for your research, please cite our paper as follows:

```
@inproceedings{cai2024dual,
  title={Dual Contrastive Graph-Level Clustering with Multiple Cluster Perspectives Alignment},
  author={Cai, Jinyu and Zhang, Yunhe and Fan, Jicong and Du, Yali and Guo, Wenzhong},
  booktitle={International Joint Conference on Artificial Intelligence},
  pages={3770--3779},
  year={2024}
}

```


