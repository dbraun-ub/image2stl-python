import numpy as np
import cv2
from stl import mesh
import argparse


def main(opt):

    # load image
    img = cv2.imread(opt['img_path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H,W = img.shape
    img = cv2.resize(img, (W//opt['scale_size'], H//opt['scale_size']))
    img = cv2.flip(img, 1)
    h,w = img.shape

    # convert image into 3d point cloud
    x,y = np.meshgrid(np.arange(w), np.arange(h))
    x *= opt['xy_res']
    y *= opt['xy_res']
    z = img / 255 * opt['nb_levels'] * 2 // 2 * opt['z_res']

    # create triangle data mesh
    data = np.zeros(4*(h*w+h+w)+2, dtype=mesh.Mesh.dtype)
    ## upper meshes
    #
    # Construct the mesh triangles
    # Add a sub-pixel at the center of four adjacent pixels to contruct the four triangles
    # Each step of the loops construct this:
    # +---+         +: pixel
    # |\ /|         *: subpixel
    # | * |         -,/,\: triangle edge
    # |/ \|
    # +---+
    it = 0
    for u in range(w-1):
        for v in range(h-1):
            # subpixel coordinates
            x0 = (x[v,u] + x[v,u+1]) / 2
            y0 = (y[v,u] + y[v+1,u]) / 2
            z0 = (z[v,u] + z[v,u+1] + z[v+1,u] + z[v+1,u+1]) / 4

            # construct the four triangles
            data['vectors'][it] = np.array([[x[v,u], y[v,u], z[v,u]],
                                            [x[v,u+1], y[v,u+1], z[v,u+1]],
                                            [x0, y0, z0]])
            data['vectors'][it+1] = np.array([[x[v,u], y[v,u], z[v,u]],
                                            [x[v+1,u], y[v+1,u], z[v+1,u]],
                                            [x0, y0, z0]])
            data['vectors'][it+2] = np.array([[x[v+1,u+1], y[v+1,u+1], z[v+1,u+1]],
                                            [x[v,u+1], y[v,u+1], z[v,u+1]],
                                            [x0, y0, z0]])
            data['vectors'][it+3] = np.array([[x[v+1,u+1], y[v+1,u+1], z[v+1,u+1]],
                                            [x[v+1,u], y[v+1,u], z[v+1,u]],
                                            [x0, y0, z0]])
            it+=4


    ## lower mesh (big rectangle)
    #
    # depth below the image surface
    d = opt['base_depth'] * opt['z_res']
    data['vectors'][it] = np.array([[x[0,0], y[0,0], -d],
                                    [x[0,0], y[-1,-1], -d],
                                    [x[-1,-1], y[-1,-1], -d]])
    it += 1
    data['vectors'][it] = np.array([[x[-1,-1], y[-1,-1], -d],
                                    [x[-1,-1], y[0,0], -d],
                                    [x[0,0], y[0,0], -d]])
    it += 1


    ## Side meshes
    #
    # along the width
    for u in range(w-1):
        data['vectors'][it] = np.array([[x[0,u], y[0,u], -d],
                                        [x[0,u], y[0,u], z[0,u]],
                                        [x[0,u+1], y[0,u+1], z[0,u+1]]])
        it += 1
        data['vectors'][it] = np.array([[x[0,u], y[0,u], -d],
                                        [x[0,u+1], y[0,u+1], -d],
                                        [x[0,u+1], y[0,u+1], z[0,u+1]]])
        it += 1
        data['vectors'][it] = np.array([[x[-1,u], y[-1,u], -d],
                                        [x[-1,u], y[-1,u], z[-1,u]],
                                        [x[-1,u+1], y[-1,u+1], z[-1,u+1]]])
        it += 1
        data['vectors'][it] = np.array([[x[-1,u], y[-1,u], -d],
                                        [x[-1,u+1], y[-1,u+1], -d],
                                        [x[-1,u+1], y[-1,u+1], z[-1,u+1]]])
        it += 1

    # along the height
    for v in range(h-1):
        data['vectors'][it] = np.array([[x[v,0], y[v,0], -d],
                                        [x[v,0], y[v,0], z[v,0]],
                                        [x[v+1,0], y[v+1,0], z[v+1,0]]])
        it += 1
        data['vectors'][it] = np.array([[x[v,0], y[v,0], -d],
                                        [x[v+1,0], y[v+1,0], -d],
                                        [x[v+1,0], y[v+1,0], z[v+1,0]]])
        it += 1
        data['vectors'][it] = np.array([[x[v,-1], y[v,-1], -d],
                                        [x[v,-1], y[v,-1], z[v,-1]],
                                        [x[v+1,-1], y[v+1,-1], z[v+1,-1]]])
        it += 1
        data['vectors'][it] = np.array([[x[v,-1], y[v,-1], -d],
                                        [x[v+1,-1], y[v+1,-1], -d],
                                        [x[v+1,-1], y[v+1,-1], z[v+1,-1]]])
        it += 1




    # save
    meshes = mesh.Mesh(data.copy())
    meshes.save(opt['save_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imag to stl converter, based on the image intensity. Works for grayscale and rgb image.")
    parser.add_argument("-i", "--img_path", type=str, help="image path")
    parser.add_argument("-s", "--save_path", type=str, help="stl save path", default="stl/mesh.stl")
    parser.add_argument("-ss", "--scale_size", type=int, default=1, help="scale image size")
    parser.add_argument("-n", "--nb_levels", type=int, default=100, help="number of intensity levels considered")
    parser.add_argument("-b", "--base_depth", type=float, default=10, help="Minimum base depth, below the surface of the image")
    parser.add_argument("-z", "--z_res", type=float, default=0.5, help="z resolution (difference between two step)")
    parser.add_argument("-xz", "--xy_res", type=float, default=1, help="pixel resolution")
    args = parser.parse_args()
    opt = vars(args)
    print(opt)

    main(opt)
