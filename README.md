<h1 align="center">
FocalPose++: Focal Length and Object Pose Estimation via Render and Compare
</h1>

<div align="center">
<h3>
<a href="http://cifkam.github.io">Martin Cífka</a>,
<a href="http://ponimatkin.github.io">Georgy Ponimatkin</a>,
<a href="http://ylabbe.github.io">Yann Labbé</a>,
<a href="http://bryanrussell.org">Bryan Russell</a>,
<a href="http://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>,
<a href="https://petrikvladimir.github.io/">Vladimír Petrík</a>,
<a href="http://www.di.ens.fr/~josef/">Josef Sivic</a>
<br>
<br>
TPAMI: IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024
<br>
<br>
<a href="https://arxiv.org/abs/2312.02985">[Paper]</a>
<a href="https://cifkam.github.io/focalpose">[Project page]</a>
</h3>
</div>


This repository contains code, models and dataset for our extension of the [FocalPose](https://github.com/ponimatkin/focalpose) method, the original paper was published on Conference on Computer Vision and Pattern Recognition, 2022.


 
## Preparing the environment and data
To prepare the environment run the following commands: 
```
conda env create -n focalpose --file environment.yaml
conda activate focalpose

git clone https://github.com/ylabbe/bullet3.git && cd bullet3 
python setup.py build
python setup.py install
```

To download the data run `bash download_data.sh`. This will download all the files 
except for the CompCars and texture datasets. For CompCars, please follow [these](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt) instructions
and download the full `.zip` archive named `CompCars.zip` into `local_data` directory. Same needs to be done for the texture dataset
which can be found at [this link](https://drive.google.com/file/d/1Xg8ODMH0k6EZLYFvQ72FF14uQ7uOitpG/view?usp=sharing). After all files are downloaded, just run
```
bash prepare_data.sh
bash preprocess_data.sh
```
This will prepare and preprocess all the files necessary for the codebase.
 
## Rendering synthetic data
The synthetic data needed for training can be generated via:
```
python -m focalpose.scripts.run_dataset_recording --config CONFIG --local --nonparametric
```
You can see all possible configs in the [run_dataset_recording.py](focalpose/scripts/run_dataset_recording.py) file. Synthetic data for Pix3D chair, CompCars and Stanford Cars 
datasets are split into multiple chunks to reduce possible rendering artifacts due to the large number of meshes. There are 21 chunks for the Pix3D chair, 10 for CompCars and 13 for Stanford Cars. 
The rendering process can be potentially sped-up by running the command without `--local` flag. This will use SLURM backend of the
[dask_jobqueue](https://jobqueue.dask.org) library. You will need to fix config of the `SLURMCluster` in the
[record_dataset.py](focalpose/recording/record_dataset.py) according to your cluster.

Alternatively, synthetic data can be downloaded [here](https://data.ciirc.cvut.cz/public/projects/2023FocalPosePP/synt_datasets/), or the original FocalPose synthetic data can be found at [this link](https://data.ciirc.cvut.cz/public/projects/2022FocalPose/synth_data/). The downloaded data should be unpacked into `local_data/synt_datasets` folder.

## Training and evaluating the models

We provide already pretrained [ML-Decoder](https://github.com/alibaba-miil/ml_decoder) classifiers located in `local_data/classifiers_ML_Decoder` folder, which appears after running the data preparation scripts. Alternatively, it can be trained via the command:
```
python -m focalpose.scripts.run_ML_Decoder_training --config pix3d-sofa-real+synth1k
```

The FocalPose model can be trained via the following command:
```
python -m focalpose.scripts.run_pose_training --config pix3d-sofa-coarse-disent-F05p
```
This particular config will train coarse model on Pix3D sofa dataset using disentangled loss and 0.5% of real-to-synth data ratio. As another example, the following command will train
refiner model on the Stanford Cars dataset with 10% of real-to-synth data ratio and using the Huber loss:
```
python -m focalpose.scripts.run_pose_training --config stanfordcars3d-refine-huber-F10p
```
We also provide an example submission scripts for [SLURM](train_slurm.sh) and [PBS](train_pbs.sh) batch systems.

To evaluate the trained coarse and refiner models run (using provided checkpoints as an example):
```
python -m focalpose.scripts.run_pose_evaluation --dataset pix3d-bed.test \
                                               --coarse-run-id pix3d-bed-coarse-disent-F05p--final \
                                               --refine-run-id pix3d-bed-refine-disent-F05p--final \
                                               --mrcnn-run-id detector-pix3d-bed-real-two-class--cvpr2022 \
                                               --classifier model-pix3d-bed-real+synth1k.ckpt
                                               --niter 15 
```
The pretrained FocalPose models are located in the `local_data/experiments` folder, which appears after running the data preparation scripts.

## Running inference on the single image
You can also directly run inference on a given image after running the data preparation scripts via:
```
python -m focalpose.scripts.run_single_image_inference --img path/to/image.jpg \
                                                       --cls class_on_image \
                                                       --niter 15 \
                                                       --topk 15 
```
This will run the inference on an image with the class manually provided to the script. The pose will be refined for 15 
iterations and the script will output top-15 model instances predicted by our instance retrieval pipeline. The ouput will consist
of images with aligned meshes, and `.txt` files containing camera matrix and camera pose.

## Citation
If you use this code in your research, please cite the following paper:

```
@article{cifka2024focalpose++,
    title={{F}ocal{P}ose++: {F}ocal {L}ength and {O}bject {P}ose {E}stimation via {R}ender and {C}ompare},
    author={C{\'\i}fka, Martin and Ponimatkin, Georgy and Labb{\'e}, Yann and Russell, Bryan and Aubry, Mathieu and Petrik, Vladimir and Sivic, Josef},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2024},
    publisher={IEEE},
    pages={1-17},
    doi={10.1109/TPAMI.2024.3475638}
}
```
