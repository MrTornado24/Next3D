# Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars

![Teaser image](./docs/rep.png)

**Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars**<br>
[Jingxiang Sun](https://mrtornado24.github.io/), [Xuan Wang](https://xuanwangvc.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), [Xiaoyu Li](https://xiaoyu258.github.io/), [Yong Zhang](https://yzhang2016.github.io/yongnorriszhang.github.io/), [Hongwen Zhang](https://hongwenzhang.github.io/), [Yebin Liu](http://www.liuyebin.com/)<br><br>
<br>https://mrtornado24.github.io/Next3D/<br>

Abstract: *3D-aware generative adversarial networks (GANs) syn-
thesize high-fidelity and multi-view-consistent facial images
using only collections of single-view 2D imagery. Towards
fine-grained control over facial attributes, recent efforts in-
corporate 3D Morphable Face Model (3DMM) to describe
deformation in generative radiance fields either explicitly
or implicitly. Explicit methods provide fine-grained expres-
sion control but cannot handle topological changes caused
by hair and accessories, while implicit ones can model var-
ied topologies but have limited generalization caused by the
unconstrained deformation fields. We propose a novel 3D
GAN framework for unsupervised learning of generative,
high-quality and 3D-consistent facial avatars from unstruc-
tured 2D images. To achieve both deformation accuracy
and topological flexibility, we propose a 3D representation
called Generative Texture-Rasterized Tri-planes. The pro-
posed representation learns Generative Neural Textures on
top of parametric mesh templates and then projects them
into three orthogonal-viewed feature planes through raster-
ization, forming a tri-plane feature representation for vol-
ume rendering. In this way, we combine both fine-grained
expression control of mesh-guided explicit deformation and
the flexibility of implicit volumetric representation. We fur-
ther propose specific modules for modeling mouth interior
which is not taken into account by 3DMM. Our method
demonstrates state-of-the-art 3D-aware synthesis quality
and animation ability through extensive experiments. Fur-
thermore, serving as 3D prior, our animatable 3D repre-
sentation boosts multiple applications including one-shot
facial avatars and 3D-aware stylization.*
