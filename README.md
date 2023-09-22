# PointNet with Self-Attention residual layers
This repository is an implementation of a final course project that studied the most influential neural architecture for Point Cloud processing called PointNet with the ambition to improve its accuracy using Self-Attention Layers from the famous Transformer architecture. A report with explanations for this study can be found [here](https://leobringer.files.wordpress.com/2023/09/ptae.pdf).


<div align="center">
  <img src="PointNet_w_SA2.png" alt="Project Description" style="width:350px;height:500px;">
</div>

&nbsp;

<p align="center">
  Some visual results of the Semantic Segmentation algorithm:
  <br>
  <img src="./gif/results_PNA.gif" alt="Image Description" width="400" height="300">
</p>

While processing Point Clouds with machine learning methods is inherently challenging due to point cloud’s irregularities and complexities, PointNet architecture has been developed to efficiently process point clouds by using multi-layer perceptrons (MLPs) and symmetric function. The architecture can be easily adapted to perform 3D recognition tasks such as classification, part segmentation, and semantic parsing, and achieve state-of-the-art performance. However, PointNet does not capture local structures induced by the metric space because of only applying elementary binary symmetric or aggregation operations, treating each points separately. This work aims to address this limitation by incorporating self-attention mechanism with positional encoding to capture local features. The extended architecture is able to achieve 3 percent increase in accuracy at 3D classification task, and detailed analysis of the architecture is provided.
