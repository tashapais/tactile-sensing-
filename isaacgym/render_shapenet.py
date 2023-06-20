import pyrender
import trimesh

# Load the 3D model from an .obj file
trimesh_scene = trimesh.load(
    '/home/tasha/Downloads/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj')

# Create a PyRender scene
scene = pyrender.Scene()

# Iterate through all geometries in the trimesh scene
for geometry in trimesh_scene.geometry.values():
    # Convert each geometry to a PyRender mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(geometry)

    # Add each mesh to the scene
    scene.add(pyrender_mesh)

# Set up the viewer to display the scene
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
