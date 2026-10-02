"""Download SIMPLE's pinned HSSD scene0 and build native MuJoCo assets.

Build dependency: usd-core==26.8. Runtime does not need USD or Isaac.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # ty: ignore[unresolved-import]

SURFACE = "/World/furniture/node_b914fb6bcc81386bfa1ff7a3eb8412b7ac581ff"
ROOT = Path(__file__).resolve().parents[1]
REVISION = "1ce0fa3956706b408df2c7c0e26b0298aa7411fd"
SCENE = "107734119_175999932"


def numbers(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def srgb(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(
        values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055
    )


def load_stage(source: Path) -> Usd.Stage:
    # The published archive has references rooted at /props and /textures.
    text = Sdf.Layer.FindOrOpen(str(source)).ExportToString()

    def asset_path(match):
        path = source.parent / match[1] / match[2]
        # The archive also references black.usd but ships Black.usd.
        if not path.exists():
            path = next(
                p for p in path.parent.iterdir() if p.name.lower() == path.name.lower()
            )
        return "@" + str(path) + "@"

    text = re.sub(
        r"@[^@]*(props|textures)/([^@]+)@",
        asset_path,
        text,
    )
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    world = UsdGeom.Xformable(stage.GetPrimAtPath("/World"))
    world.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bound = bounds.ComputeWorldBound(stage.GetPrimAtPath(SURFACE))
    axes = np.asarray(bound.GetMatrix().ExtractRotationMatrix())
    half_extent = np.asarray(bound.GetRange().GetSize()) / 2
    # SIMPLE averages the OBB corners along positive local Y (the table top).
    surface_top = np.asarray(bound.ComputeCentroid()) + axes[1] * half_extent[1]
    offset = -surface_top + np.array([0.25, 0, 0])
    offset[2] = 0  # G1 SONIC preserves the source floor's vertical position.
    stage.GetPrimAtPath("/World").GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(*offset)
    )
    # Isaac's scene loader overrides the authored invisible ceiling to visible.
    stage.GetPrimAtPath("/World/ceilings").GetAttribute("visibility").Set("inherited")
    stage.GetPrimAtPath(SURFACE).GetAttribute("visibility").Set("invisible")
    return stage


def material(prim, assets, output, material_ids):
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    path = str(bound.GetPath())
    if path in material_ids:
        return material_ids[path]
    name = f"material_{len(material_ids):03d}"
    shader = next(
        UsdShade.Shader(p)
        for p in Usd.PrimRange(bound.GetPrim())
        if p.IsA(UsdShade.Shader)
        and UsdShade.Shader(p).GetSourceAssetSubIdentifier("mdl") == "gltf_material"
    )
    color = shader.GetInput("base_color_factor").Get()
    alpha = float(shader.GetInput("base_alpha").Get())
    roughness = float(shader.GetInput("roughness_factor").Get())
    metallic = float(shader.GetInput("metallic_factor").Get())
    transmission = shader.GetInput("transmission_factor")
    if transmission:
        alpha *= 1 - 0.65 * float(transmission.Get())
    attributes = {
        "name": name,
        "rgba": numbers([*srgb(color), alpha]),
        "specular": f"{0.15 + 0.45 * metallic:.5g}",
        "shininess": f"{(1 - roughness) ** 2:.5g}",
    }
    uv_scale = np.ones(2)
    texture_input = shader.GetInput("base_color_texture")
    if texture_input and texture_input.HasConnectedSource():
        source = texture_input.GetConnectedSource()[0]
        lookup = UsdShade.Shader(source.GetPrim())
        uv_scale = np.asarray(lookup.GetInput("scale").Get())
        texture_path = Path(lookup.GetInput("texture").Get().path)
        texture_name = f"texture_{len(material_ids):03d}"
        texture_file = texture_name + ".png"
        with Image.open(texture_path) as image:
            image.convert("RGB").save(output / "textures" / texture_file)
        ET.SubElement(
            assets,
            "texture",
            name=texture_name,
            type="2d",
            file="textures/" + texture_file,
        )
        attributes.update(texture=texture_name, texuniform="false")
    ET.SubElement(assets, "material", attributes)
    material_ids[path] = name, uv_scale
    return name, uv_scale


def corner_values(values, interpolation, vertex_ids, face_counts):
    if interpolation == "faceVarying":
        return np.asarray(values)
    if interpolation in ("vertex", "varying"):
        return np.asarray(values)[vertex_ids]
    if interpolation == "uniform":
        return np.repeat(np.asarray(values), face_counts, axis=0)
    return np.repeat(np.asarray(values), len(vertex_ids), axis=0)


def export_mesh(mesh, transform, face_ids, uv_scale, destination):
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    points = points @ transform[:3, :3] + transform[3, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if np.any(counts != 3):
        raise ValueError(f"Expected triangles in scene0: {mesh.GetPath()}")
    vertex_ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    triangles = vertex_ids.reshape(-1, 3)[face_ids]
    corners = np.arange(len(vertex_ids)).reshape(-1, 3)[face_ids]
    if (np.linalg.det(transform[:3, :3]) < 0) != (
        mesh.GetOrientationAttr().Get() == "leftHanded"
    ):
        triangles = triangles[:, ::-1]
        corners = corners[:, ::-1]
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None and len(normals):
        normals = corner_values(
            normals, mesh.GetNormalsInterpolation(), vertex_ids, counts
        )
        normals = normals @ np.linalg.inv(transform[:3, :3]).T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    texcoords = None
    if uv and uv.HasValue():
        texcoords = (
            corner_values(
                uv.ComputeFlattened(), uv.GetInterpolation(), vertex_ids, counts
            )
            * uv_scale
        )
    with destination.open("w") as stream:
        np.savetxt(stream, points, fmt="v %.8g %.8g %.8g")
        if texcoords is not None:
            np.savetxt(stream, texcoords, fmt="vt %.8g %.8g")
        if normals is not None and len(normals):
            np.savetxt(stream, normals, fmt="vn %.8g %.8g %.8g")
        for vertices, indices in zip(triangles + 1, corners + 1):
            face = []
            for vertex, index in zip(vertices, indices):
                if normals is not None and len(normals):
                    face.append(
                        f"{vertex}/{index if texcoords is not None else ''}/{index}"
                    )
                elif texcoords is not None:
                    face.append(f"{vertex}/{index}")
                else:
                    face.append(str(vertex))
            stream.write("f " + " ".join(face) + "\n")
    return points.min(axis=0), points.max(axis=0), len(triangles)


def convert(source: Path, output: Path):
    stage = load_stage(source)
    (output / "meshes").mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    root = ET.Element("mujoco", model="HSSD scene0 visuals")
    ET.SubElement(root, "compiler", angle="radian")
    assets = ET.SubElement(root, "asset")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="room")
    transforms = UsdGeom.XformCache()
    material_ids = {}
    manifest = []
    triangle_count = 0
    for prim in stage.Traverse():
        if (
            not prim.IsA(UsdGeom.Mesh)
            or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        groups = [(prim, np.arange(len(mesh.GetFaceVertexCountsAttr().Get())))]
        subsets = UsdGeom.Subset.GetGeomSubsets(
            mesh, UsdGeom.Tokens.face, "materialBind"
        )
        if subsets:
            assigned = set()
            groups = []
            for subset in subsets:
                indices = np.asarray(subset.GetIndicesAttr().Get())
                groups.append((subset.GetPrim(), indices))
                assigned.update(indices.tolist())
            remaining = np.asarray(
                [
                    i
                    for i in range(len(mesh.GetFaceVertexCountsAttr().Get()))
                    if i not in assigned
                ]
            )
            if len(remaining):
                groups.append((prim, remaining))
        for bound_prim, face_ids in groups:
            if not len(face_ids):
                continue
            name = f"mesh_{len(manifest):03d}"
            mat, uv_scale = material(bound_prim, assets, output, material_ids)
            low, high, faces = export_mesh(
                mesh,
                np.asarray(transforms.GetLocalToWorldTransform(prim)),
                face_ids,
                uv_scale,
                output / "meshes" / (name + ".obj"),
            )
            triangle_count += faces
            ET.SubElement(
                assets,
                "mesh",
                name=name,
                file="meshes/" + name + ".obj",
                inertia="shell",
            )
            ET.SubElement(
                body,
                "geom",
                name=name,
                type="mesh",
                mesh=name,
                material=mat,
                contype="0",
                conaffinity="0",
                density="0",
                group="2",
            )
            manifest.append(
                {
                    "mesh": name,
                    "source": str(prim.GetPath()),
                    "faces": faces,
                    "bounds": [low.tolist(), high.tolist()],
                }
            )
    ET.indent(root)
    ET.ElementTree(root).write(output / "room.xml", encoding="unicode")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Converted {len(manifest)} meshes, {triangle_count} triangles, {len(material_ids)} materials"
    )


if __name__ == "__main__":
    print("Fetching pinned HSSD archive...", flush=True)
    archive = hf_hub_download(
        repo_id="USC-PSI-Lab/SIMPLE",
        repo_type="dataset",
        filename=f"scenes_hssd_{SCENE}.zip",
        revision=REVISION,
        cache_dir=ROOT / "artifacts/hssd",
    )
    output = ROOT / "assets/hssd/scene0"
    with TemporaryDirectory(prefix="rlora-hssd-") as directory:
        with ZipFile(archive) as source:
            source.extractall(directory)
        print("Converting room to native MuJoCo assets...", flush=True)
        convert(Path(directory) / f"scenes/hssd/{SCENE}/{SCENE}.usd", output)
    print(f"Room ready: {output}")
