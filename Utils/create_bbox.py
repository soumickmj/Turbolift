import random

import matplotlib.colors as mcolors
# import napari #required for 3D viz
import numpy as np
# import vispy.color #required for 3D viz
from matplotlib import patches
from matplotlib import pyplot as plt
from skimage import measure


def seg2bbx(seg):
    """Get the coordinates of the bounding boxes from a segmentation mask

    Parameters
    ----------
    seg : nD array containing the segmentation mask

    Returns
    -------
    bboxes : List[Tuple]
        Returns a list of tuple, where each element corresponds to each of the bounding boxes
        Each of the elements is a tuple, ordered as: (min_dim0, min_dim1, min_dimN, max_dim0, max_dim1, max_dimN), where "N" is the number of dims
        For 2D: (min_row, min_col, max_row, max_col)
        For 3D: (min_row, min_col, min_slice, max_row, max_col, max_slice)
    centroids : List[Tuple]
        Returns a list of tuple, where each element corresponds to the centroids of each of the bounding boxes
        Each of the elements is a tuple, ordered as: (cent_dim0, cent_dim1, min_dimN), where "N" is the number of dims
        For 2D: (cent_dim0, cent_dim1)
        For 3D: (cent_dim0, cent_dim1, cent_dim2)
    """
    label_img = measure.label(seg)
    regions = measure.regionprops(label_img)
    areas = {
        "actual": [regions[i].area for i in range(len(regions))],
        "bbox_areas": [regions[i].bbox_area for i in range(len(regions))],
        "convex_areas": [regions[i].convex_area for i in range(len(regions))],
    }

    bboxes = [regions[i].bbox for i in range(len(regions))]
    centroids = [regions[i].centroid for i in range(len(regions))]
    return bboxes, centroids, areas


def make_bbox3D(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (N)
        List of the extents of the bounding boxes for each of the N regions.
        Each of the elements should be ordered: [min_row, min_column, min_slice, max_row, max_column, max_slice]

    Returns
    -------
    bbox_cube : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    bbox_extents = np.array(bbox_extents).transpose()

    minx = bbox_extents[0]
    miny = bbox_extents[1]
    minz = bbox_extents[2]

    maxx = bbox_extents[3]
    maxy = bbox_extents[4]
    maxz = bbox_extents[5]

    bbox_cubes = np.array([[minx, miny, minz], [maxx, miny, minz], [maxx, maxy, minz], [minx, maxy, minz], [minx, miny, minz],
                          [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz], [
                              maxx, maxy, minz], [maxx, maxy, maxz],
                          [maxx, miny, maxz], [minx, miny, maxz], [minx, maxy, maxz], [
                              minx, maxy, minz], [minx, maxy, maxz], [maxx, maxy, maxz],
                          [maxx, maxy, minz], [minx, maxy, minz], [minx, maxy, maxz]])

    bbox_cubes = np.moveaxis(bbox_cubes, 2, 0)
    return bbox_cubes


def make_bbox_3Dto2D(bboxes, slice):
    onlySlice = []
    for bbox in bboxes:
        (min_row, min_col, min_slice, max_row, max_col, max_slice) = bbox
        if min_slice <= slice <= max_slice:
            onlySlice.append((min_row, min_col, max_row, max_col))
    return onlySlice


def view_bbox2D(seg, bboxes, slice):
    bbox_rects = make_bbox_3Dto2D(bboxes, slice)
    fig, ax = plt.subplots()
    ax.imshow(seg[..., slice], cmap=plt.cm.gray)
    for bbox in bbox_rects:
        (min_row, min_col, max_row, max_col) = bbox
        height = max_row - min_row
        width = max_col - min_col
        rect = patches.Rectangle((min_col, min_row), width,
                                 height, linewidth=1, edgecolor=random.choices([c for c in mcolors.CSS4_COLORS.keys() if len(
                                     c) > 1 and ("white" not in c) and ("black" not in c)], k=1)[0], facecolor='none')
        ax.add_patch(rect)
    plt.show()


# def view_bbox3D(seg, bboxes):
#     bbox_cubes = make_bbox3D(bboxes)
#     viewer = napari.Viewer(ndisplay=3)
#     new_layer = viewer.add_image(seg, rgb=False)
#     shapes_layer = viewer.add_shapes(
#         bbox_cubes,
#         shape_type="path",
#         face_color='transparent',
#         edge_color=random.choices([c for c in vispy.color.get_color_names() if len(
#             c) > 1 and ("white" not in c) and ("black" not in c)], k=len(bbox_cubes)),
#         name='bounding box',
#     )
#     napari.run()
