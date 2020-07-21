Source code for "PointTriNet: Learned Triangulation of 3D Point Sets", by Nicholas Sharp and Maks Ovsjanikov at ECCV 2020

**IN PROGRESS** This repository is currently being populated, check back in over the next few days.


## Dependencies

Depends on `pytorch`, `torch-scatter`, and `polyscope`. The code is pretty standard, and there shouldn't any particularly strict version requirements on these depednencies; any recent version should work fine. 

For reproducibilty, a `requirements.txt` is included, which can be used to construct a suitable conda environment like
```sh
TODO conda stuff
```

The codebase should work fine on CPU or cuda, and does not require an special precompiled modules. As usual, you may find the pytorch neural nets to be unacceptably slow on CPU.

## Example: Generate a mesh


The script `main_generate_mesh.py` applies a trained model to triangulate a point set. A set of pretrained weights are included in `saved_models/`

```sh
python src/main_generate_mesh.py path/to/points.ply  --weights=saved_model/
```

This script has the following arguments

| flag | purpose | arguments |
| :------------- |:------------- | :-----|
| `--noGUI` | Do not show the GUI, just process options and exit | |
| `--flipDelaunay` | Flip edges to make the mesh intrinsic Delaunay | |
| `--refineDelaunay` | Refine and flip edges to make the mesh intrinsic Delaunay and satisfy angle/size bounds | |

This script automatically applies the PointTriNet networks over patches of the shape, allowing it to scale to very large point sets without memory concerns.  The option `--patch-size` adjusts the size of these patches, smaller patches limit memory usage at the cost of a little speed (this is a performance optimization only, results of the method are unaffected).  This simple strategy works great for forward inference, but doesn't help if gradients are needed for a large point set, as PyTorch will retain the whole computation graph regardless---see the training scripts for more advanced patch management.



## Example: Integrating with code

If you want to integrate PointTriNet into your own codebase, the `PointTriNet_Mesher` from `point_tri_net.py` encapsulates all the functionality of the method. It's a `torch.nn.Module`, so you can make it a member of other modules, load weights, etc.

To create the model, load weights, and triangulate a point set, just call:

TODO RUN THIS CODE

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

## Example: Train the model

First, generate data
