# HSSD asset license and attribution

The meshes and textures in `scene0/` are adapted from the
[Habitat Synthetic Scenes Dataset (HSSD)](https://3dlg-hcvc.github.io/hssd/),
scene `107734119_175999932`, distributed in
[USC-PSI-Lab/SIMPLE](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE).

HSSD is released under [Creative Commons Attribution–NonCommercial 4.0
International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The [legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
applies to these derived assets; SIMPLE's MIT code license does not replace it.

HSSD authors: Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan
Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and
Manolis Savva.

Changes: converted the SIMPLE USD scene into MuJoCo OBJ/MJCF; baked scene and
mesh transforms; hid the selected source tea table; converted diffuse textures
into RGB PNGs; approximated glTF/MDL surface properties with MuJoCo materials.
