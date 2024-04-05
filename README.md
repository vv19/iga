# Implicit Graph Alignment (IGA)

Code for the paper: "Few-Shot In-Context Imitation Learning via Implicit Graph Alignment" (Published at CoRL 2023).
[Project Webpage](https://www.robot-learning.uk/implicit-graph-alignment), [Paper](https://arxiv.org/pdf/2310.12238.pdf)

<p align="center">
<img src="./media/overview.gif" alt="drawing" width="700"/>
</p>

Implicit Graph Alignment (IGA) is a method for few-shot in-context imitation learning that aligns the objects in a way that is consistent with provided the demonstrations (context) by optimising their alignments in the graph space.

<p align="center">
<img src="./media/graph_optim.gif" alt="drawing" width="600"/>
</p>

## Setup

**Clone this repo**

```
git clone https://github.com/vv19/iga.git
cd iga
```

**Create conda environment**

```
conda env create -f environment.yml
conda activate iga_env
pip install -e .
```

## Quick Demo

We provide a quick demonstration of the optimisation procedure of alignments between different objects. Demo data contains point clouds from simulation and ones captured in the real world. By running the demo, you will see the initial "random" alignments between the two objects, the optimisation procedure based on the provided demonstrations (context), and the produced alignment between the objects. During real-world deployment, these point clouds would be captured using depth cameras.

**Download pre-trained weights and demo data**

```
cd iga
./scripts/download_weights.sh
./scripts/download_demo_data.sh
```

**Run the demo**

```
python inference.py
```

For more optimisation and visualisation options see `python inference.py --help`.

For example, to visualise how the optimisation looks like directly in the graph space, run:

```
python inference.py --visualise_optimisation='graph'
```

## Usage
**Run with your own data**

To use IGA for your own data, specify the directory (`python inference.py --data_dir=... --real_data=True`) where the data is stored.
The data, saved as pickle files (e.g. `sample_0.pkl`, `sample_1.pkl`, ...), should be stored in the following format:

```
sample = {
 'pcds_a': pcds_a,  # list of point clouds (np.arrays) for the grasped object, None if no object is grasped.
 'pcds_b': pcds_b,  # list of point clouds (np.arrays) for the target object.
}
```

The first point cloud in `pcds_a` (and `pcds_b`) is used as the live point cloud, the rest are used for the context. 

The models are trained on **complete point clouds expressed in the robot's end-effector frame -- objects are grasped in a similar way**. 
Please ensure that the point clouds are in the same format.

**Using IGA for robot experiments**

To use IGA for your robot experiments, you can adapt the `inference.py`.
`iga.get_transformation(...)` returns the transformation matrix `T_e_e_new` that needs to be applied to the end-effector to align the objects.
Compute the new end-effector pose as follows:
```
T_w_e_new = T_w_e_current @ T_e_e_new
```

If the performance is not satisfactory, you can try to adjust the optimisation parameters, increase the context length,
or fine-tune the models.

## Training / Fine-Tuning

If you want to train the models from scratch or fine-tune them, you can use the `train.py` script.
E.g. to fine-tune the rotation model on the demo data, run:

```
python train_ebm.py \
 --run_name='emb_rot_finetuned' \
 --data_root='./demo_data' \
 --data_root_val='./demo_data' \
 --record=True \
 --reprocess=True \
 --mode='rot' \
 --num_samples=15 \
 --model_path='./checkpoints/ebm_rot.pt'
```

data_dir and data_dir_val should contain the same format as described above, with all point clouds aligned in a consistent way.

Training parameters are defined in `/configs/ebm_rot.py` and `/configs/ebm_trans.py` and can be adjusted as needed.

# Citing

If you find our paper interesting or this code useful in your work, please cite our paper:

```

@inproceedings{vosylius2023few,
title={Few-Shot In-Context Imitation Learning via Implicit Graph Alignment},
author={Vosylius, Vitalis and Johns, Edward},
booktitle={Conference on Robot Learning},
year={2023}
}
```
