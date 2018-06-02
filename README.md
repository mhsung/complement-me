## ComplementMe: Weakly-Supervised Component Suggestions for 3D Modeling
<a href="http://cs.stanford.edu/~mhsung" target="_blank">Minhyuk Sung</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://vova.kim" target="_blank">Vladimir G. Kim</a>, <a href="https://www.cse.iitb.ac.in/~sidch/" target="_blank">Siddhartha Chaudhuri</a>, and <a href="https://geometry.stanford.edu/member/guibas/" target="_blank:">Leonidas Guibas</a><br>
Siggraph Asia 2017

![teaser](https://mhsung.github.io/assets/images/component-assembly/teaser.png)

[[Project]](https://mhsung.github.io/component-assembly.html) | [[arXiv]](https://arxiv.org/abs/1708.01841)

### Citation
```
@article{Sung:2017,
  author = {Sung, Minhyuk and Su, Hao, and Kim, Vladimir G. and Chaudhuri, Siddhartha
    and Guibas, Leonidas},
  title = {ComplementMe: Weakly-Supervised Component Suggestions for 3D Modeling},
  Journal = {ACM Transactions on Graphics (Proc. of SIGGRAPH Asia)}, 
  year = {2017}
}
```

### Introduction
ComplementMe is a neural network framework for suggesting complementary components and their placement for an incomplete 3D part assembly. The component retrieval is performed by two neural networks called *embedding* and *retrieval* networks; the first indexes parts by mapping them to a low-dimensional feature space, and the second maps partial assemblies to appropriate complements. These two networks are *jointly* trained on *unlabeled* data obtained from public online repositories without relying on consistent part segmentations or labels. The retrieval network predicts a *probability distribution* over the space of part embeddings to deal with ambiguities of the multiple complementary components. The placement is performed by a separate network called *placement* network, which predicts a coordinates of the newly added component.

### Data download
The ShapeNet model component and semantic part data are available on our [project website](https://mhsung.github.io/component-assembly.html).

### Requirements
- Numpy (tested with ver. 1.13.1)
- TensorFlow (tested with ver. 1.0.1)

### Acknowledgements
The files in [utils](utils) are directly brought from the [PointNet](https://github.com/charlesq34/pointnet).

### License
This code is released under the MIT License. Refer to [LICENSE](LICENSE) for details.

### To-Do
- [ ] Script files reproducing results in the paper.
