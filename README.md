# DeepNNK
 - Non parametric, polytope interpolation framework for use with deep learning models
 
 Python source code for paper: [DeepNNK: Explaining deep models and their generalization using polytope interpolation](https://arxiv.org/abs/2007.10505)
 ### Code requirements
 The code is tested for python3.6. Use with python2 will require some minor updates in the packages installed and source code.
 - `requirements.txt` contains pip packages to be installed. 
 - Packages listed assume no GPU availability. Use `faiss-gpu` 
 and `tensorflow-gpu` for use with GPU.
 
 ### Reproducing results
 - `run_script.bash` shows a simple set of commands needed to train, test and calibrate a regularized model on CIFAR-10 dataset.
 - Modify `regularize` and `layer_size` flags to create different models.
 - Set `mode` flag to `plot` to obtain predictions and associated NNK neighbors. Image ID's to be plotted is to be updated in `main.py`
 Plots obtained for model selection experiment presented in paper can be obtained by running `python overfitting_study.py`
 after setting up the appropriate paths to the models in code.
 - The source code contains the models trained and used for paper in `logs` directory.
    
 ## Citing this work
 ```
@article{shekkizhar2020deepnnk,
  title={DeepNNK: Explaining deep models and their generalization using polytope interpolation},
  author={Shekkizhar, Sarath and Ortega, Antonio},
  journal={arXiv preprint arXiv:2007.10505},
  year={2020}
}
```