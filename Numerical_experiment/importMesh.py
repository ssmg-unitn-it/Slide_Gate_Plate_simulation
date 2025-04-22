# -----------------------------------------------------------------70
# Import necessary packages
# -----------------------------------------------------------------70
import fenics as fe
import numpy as np
import meshio
import os
import sys

#%% ---------------------------------------------------------------70
# Auxiliary functions
# -----------------------------------------------------------------70

def fileprint(string, meshfolder, meshfilename):
    print(string)
    with open( meshfolder + meshfilename + ".md", "a") as file:
        file.writelines( "\n" + string)


def getMeshInfo(meshfolder, meshfile, identifier = "cell", cell_geom = "triangle"):
    """Import the mesh from Abaqus and convert the mesh with Meshio

    - use a specific keywork to identify cell sets in the mesh creation
      and pass this keywork to the function as identifier

    - specify whether to parse for triangle or tetra:
        - 2D case --> cell_geom = "triangle"
        - 3D case --> cell_geom = "tetra"

    reference:
    https://fenicsproject.discourse.group/t/definition-of-subdomain-by-element-sets/6440
    """

    # Take the mesh file name without extension
    meshfilename = meshfile.rsplit(".")[0]

    # read the mesh whatever the format
    pre_mesh = meshio.read(meshfolder + meshfile)

    # import the cells sets
    cell_sets = pre_mesh.cell_sets_dict

    # import the cells
    pre_cells = pre_mesh.get_cells_type(cell_geom)

    # number of cells
    n_cells = len(pre_cells)

    # initialize a matrix to store sets marks for cell data (materials)
    cell_data = np.zeros(n_cells)

    # initialize a matrix to store sets marks for facet data (boundaries)
    boundaries_data = np.zeros(n_cells)

    # initialize marker counter
    mrk = 0

    # initialize marker dictionary
    mrk_dict = {}

    # loop through the dictionary items of cell_sets
    # the key is the set name, the value is the sets cells (type and coordinates)
    for marker, set in sorted(cell_sets.items()):
        if "mesh" not in marker:
            # Increse the marker for every set
            mrk += 1

            # Store set name and corresponding marker
            mrk_dict[marker] = mrk

            # loop through the types and values for a specific key
            for type, entities in set.items():
                # check for cell type
                if type == cell_geom:
                    # Store in cell_data a marker with increasing int to be visualized in Paraview
                    boundaries_data[entities] = int(mrk)

                    # identify cell markers and store them in a list
                    if identifier in marker:
                        cell_data[entities] = int(mrk)

    return pre_mesh, cell_data, boundaries_data, mrk, mrk_dict




def convertMesh(meshfolder, meshfile, identifier = "cell", cell_geom = "triangle"):

    (pre_mesh, cell_data, boundaries_data, mrk, mrk_dict) = getMeshInfo(meshfolder, meshfile, identifier, cell_geom)

    # import the cells
    pre_cells = pre_mesh.get_cells_type(cell_geom)

    # number of cells
    n_cells = len(pre_cells)

    # Take the mesh file name without extension
    meshfilename = meshfile.rsplit(".")[0]

    # Print the total number of cells
    fileprint(f"Total number of cells in mesh = {n_cells}", meshfolder, meshfilename)

    # Print the total number of markers
    fileprint(f"Max marker index: {mrk:2d}", meshfolder, meshfilename)

    # Print set name and corresponding marker
    for key in mrk_dict.keys():
        fileprint(f"Set: {key} -> marker {mrk_dict[key]}", meshfolder, meshfilename)

    # Translate the mesh in xdmf format bringing the marker info
    post_mesh = meshio.Mesh(points=pre_mesh.points,
                            cells={cell_geom: pre_cells},
                            cell_data={"identifier": [cell_data]} )

    # Take the mesh file name without extension
    meshfilename = meshfile.rsplit(".")[0]

    # Adjust the filename and output the mesh file in xdmf format
    meshio.write( meshfolder + meshfilename + ".xdmf", post_mesh)




def readMesh(meshfolder, meshfile, identifier = "cell", cell_geom = "triangle", flush_files_flag=0):
    # -----------------------------------------------------------------70
    # Convert solid Element sets in surface boundary Facets Sets
    # reference
    # https://fenicsproject.discourse.group/t/how-to-define-boundary-conditions-on-an-irregular-geometry/2240/18
    # -----------------------------------------------------------------70

    # Take the mesh file name without extension
    meshfilename = meshfile.rsplit(".")[0]

    # check if mesh has already been converted in xdmf
    xdmf_exists_flag = 0
    for file in os.listdir(meshfolder):
        if file == meshfilename + ".xdmf":
            xdmf_exists_flag = 1

    # convert the mesh if necessary
    if xdmf_exists_flag == 0 or flush_files_flag:
        convertMesh(meshfolder, meshfile, cell_geom = cell_geom)

    # import the mesh in FEniCS
    with fe.XDMFFile( meshfolder + meshfilename + ".xdmf") as xdmf:
        mesh = fe.Mesh()
        xdmf.read(mesh)
        mesh.init()

    # Define the whole boundary
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    # initialize marker for the whole boundary
    boundaries.set_all(0)

    # define the whole domain
    materials = fe.MeshFunction("size_t", mesh, mesh.topology().dim())

    # initialize marker for the whole domain
    materials.set_all(0)

    # initialize mesh to ask for facets
    cell_to_facet = mesh.topology()(mesh.topology().dim(), mesh.topology().dim()-1)

    # get cell_data from meshio
    (pre_mesh, cell_data, boundaries_data, mrk, mrk_dict) = getMeshInfo(meshfolder, meshfile, identifier, cell_geom)


    # loop through cells
    for cell in fe.cells(mesh):

        # mark the cell with info stored in cell_data
        materials.set_value(cell.index(), int(cell_data[cell.index()]))

        # loop through the facets of a cell
        for face in cell_to_facet(cell.index()):
            # extract Facet info with Dolfin internal method
            facet = fe.Facet(mesh, face)
            # mark the facet with the info stored in boundaries_data
            if facet.exterior():
                boundaries.set_value(facet.index(), int(boundaries_data[cell.index()]))

    # https://www.karlin.mff.cuni.cz/~hron/fenics-tutorial/elasticity/doc.html
    return mesh, boundaries, materials


#%% ---------------------------------------------------------------70
# Debugging check
# -----------------------------------------------------------------70

if __name__ == "__main__":

    meshfolder = "./Meshes/"

    meshfile = sys.argv[1]

    mesh, boundaries, materials = readMesh(meshfolder, meshfile, cell_geom = "triangle", flush_files_flag = 1)

    CHECK = 1

    if CHECK == 1:
        fe.XDMFFile( meshfolder + "check_boundaries.xdmf" ).write(boundaries)
        fe.XDMFFile( meshfolder + "check_materials.xdmf" ).write(materials)


# https://www.karlin.mff.cuni.cz/~hron/fenics-tutorial/appendix/doc.html#mesh-import
# https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/mesh/refinement/refine.html
# refined_mesh = fe.refine(mesh)
