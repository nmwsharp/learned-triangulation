Source code & pretrained model for "[PointTriNet: Learned Triangulation of 3D Point Sets](https://nmwsharp.com/research/learned-triangulation/)", by [Nicholas Sharp](https://nmwsharp.com/) and [Maks Ovsjanikov](http://www.lix.polytechnique.fr/~maks/) at ECCV 2020.

- PDF: [link](https://nmwsharp.com/media/papers/learned-triangulation/learned_triangulation.pdf)
- Project: [link](https://nmwsharp.com/research/learned-triangulation/)
- Talk: [link](https://www.youtube.com/watch?v=PoNT0u_wz4Y)


![demo gif](https://github.com/nmwsharp/learned-triangulation/blob/master/teaser.gif)

## Example: Generate a mesh


The script `main_generate_mesh.py` applies a trained model to triangulate a point set. A set of pretrained weights are included in `saved_models/`

```sh
python src/main_generate_mesh.py saved_models/model_state_dict.pth path/to/points.ply
```

Check out the `--help` flag on the script for arguments. In particular, the script can either take a point cloud directly as input, or take a mesh as input and uniformly sample points with `--sample_cloud`.

Note that by default, the script opens up a GUI (using [Polyscope](http://polyscope.run/)) to show results. To skip the GUI and just write out the resulting mesh, use:

```sh
python src/main_generate_mesh.py path_to_your_cloud_or_mesh.ply --output result
```

## Example: Integrating with code

If you want to integrate PointTriNet in to your own codebase, the `PointTriNet_Mesher` from `point_tri_net.py` encapsulates all the functionality of the method. It's a `torch.nn.Module`, so you can make it a member of other modules, load weights, etc.

To create the model, load weights, and triangulate a point set, just call:

```python

model = PointTriNet_Mesher()
model.load_state_dict(torch.load(some_path))
model.eval()

samples = # your (B,V,3) torch tensor of point positions

with torch.no_grad():
  candidate_triangles, candidate_probs = model.predict_mesh(samples)
  # candidate_triangles is a (B, F, 3) index tensor, predicted triangles
  # candidate_probs is a (B, F) float tensor of [0,1] probabilities for each triangle

  # You are probably interested in only the high-probability triangles. For example,
  # get the high-probability triangles from the 0th batch entry like
  b = 0
  prob_thresh = 0.9
  high_prob_faces = candidate_triangles[b, candidate_probs[b,:] > prob_thresh, :]


```

## Example: Generate data & train the model

**Prerequisite**: a collection of shapes to train on; we use the training set (all classes) of ShapeNet v2, which you can download on your own. Note that we _do not_ train PointTriNet to match the triangulation of existing meshes, we're just using meshes as a convenient data source from which to sample point cloud patches.

**Step 1** Sample point cloud patches as training (and validation) data

```shell
python src/generate_local_points_dataset.py --input_dir=/path/to/train_meshes/ --output_dir=data/train/ --n_samples=20000

python src/generate_local_points_dataset.py --input_dir=/path/to/val_meshes/ --output_dir=data/val/ --n_samples=5000
```

**Step 2** Train the model

```sh
python src/main_train_model.py
```

With default parameters, this will train for 3 epochs on the dataset above, using < 8GB gpu memory and taking ~6hrs on an RTX 2070 GPU. Checkpoints will be saved in `./training_runs`, along with tensorboard logging.

Note that this script has paths at the top relative to the expected directory layout of this repo. If you want to use a different directory layout, you can update the paths.

## Dependencies

Depends on `pytorch`, `torch-scatter`, `libigl`, and `polyscope`, along with some other typical numerical components. The code is pretty standard, and there shouldn't be any particularly strict version requirements on these dependencies; any recent version should work fine.

For completeness, an `environment.yml` file is included (which is a superset of the required packages).

## Citation

If this code contributes to academic work, please cite:

```bib
@inproceedings{sharp2020ptn,
  title={"PointTriNet: Learned Triangulation of 3D Point Sets"},
  author={Sharp, Nicholas and Ovsjanikov, Maks},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={},
  year={2020}
}
```

