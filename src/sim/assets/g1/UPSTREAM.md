# G1 Dex3 asset

`g1_29dof_with_hand.xml` and its referenced meshes come from
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl)
at commit `4141c34280abb67c82e115342a8720f4a83d750d`:

```text
gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml
gear_sonic/data/robot_model/model_data/g1/meshes/
```

The XML actuator block was removed because MJLab creates the body and hand
position actuators from `src/sim/config.py`.

The source repository licenses its source code under Apache-2.0.
