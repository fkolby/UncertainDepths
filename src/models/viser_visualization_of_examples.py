"""GUI basics

Examples of basic GUI elements that we can create, read from, and write to."""

import os
import time
from copy import deepcopy

import kornia
import numpy as np
import torch
from torchvision.transforms import Resize
from pprint import pprint
from typing import List

import viser


def get_intrinsics(H: float, W: float) -> torch.Tensor:  # Inspired by gradio demo of ZoeDepth
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return torch.tensor(
        [[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
    )


def img_dicts(picture_folders: List[str]) -> dict[str, dict[str, List[torch.Tensor]]]:
    """Generates a dictionary to be indexed by viser tool for 3D visualizations.
    Takes images from picture folders"""

    d = {"depth": [], "img": [], "preds": [], "ScaledUncertainty": [], "std_dev": [], "diff": []}
    d = {
        "Posthoc_Laplace": deepcopy(d),
        "Online_Laplace": deepcopy(d),
        "Ensemble": deepcopy(d),
        "Dropout": deepcopy(d),
        "ZoeNK": deepcopy(d),
    }

    path_to_folder = "/home/jbv415/UncertainDepths/src/models/outputs/images/"

    for picfold in picture_folders:
        path_to_pics = os.path.join(path_to_folder, picfold)

        visuals = os.listdir(path_to_pics)
        visuals.sort()
        print(visuals)
        for el in visuals:
            if el[-4:] != ".npy":  # or len(el.split("_")) != 4:
                continue
            print(el)

            idx_inc = 0
            im_type = el.split("_")[1]
            if im_type == "std":
                im_type = "std_dev"
                idx_inc = 1

            model = el.split("_")[2 + idx_inc]
            if model == "Posthoc" or model == "Online":
                # then it looks like "np_imtype_Posthoc_laplace_imnum.npy"
                model = "_".join(el.split("_")[2 + idx_inc : 4 + idx_inc])
            print(model)
            print(im_type)
            print(el)
            print(torch.tensor(np.load(os.path.join(path_to_pics, el))).shape)
            if im_type == "img":
                print(torch.tensor(np.load(os.path.join(path_to_pics, el))).shape)
                if model == "ZoeNK":
                    d[model][im_type] = d[model][im_type] + [
                        Resize((352, 704))(
                            torch.tensor(np.load(os.path.join(path_to_pics, el))) / 100.0
                        ).numpy(force=True)
                    ]
                else:
                    d[model][im_type] = d[model][im_type] + [
                        Resize((352, 704))(
                            torch.tensor(np.load(os.path.join(path_to_pics, el)))
                        ).numpy(force=True)
                    ]
                """ elif im_type == "depth":  # Resize, so preds/depths_dict can be color-corrected
                d[model][im_type] = d[model][im_type] + [
                    Resize((352, 704))(
                        torch.unsqueeze(torch.tensor(np.load(os.path.join(path_to_pics, el))), dim=0)
                    )
                    .squeeze()
                    .numpy(force=True)
                ] """
            elif im_type in [
                "std_dev",
                "ScaledUncertainty",
            ]:  # Resize, so preds/depths can be color-corrected
                if model == "ZoeNK":
                    d[model][im_type] = d[model][im_type] + [
                        Resize((352, 704))(
                            torch.unsqueeze(
                                torch.tensor(np.load(os.path.join(path_to_pics, el))) / 255.0, dim=0
                            )
                        ).numpy(force=True)
                    ]
                else:
                    d[model][im_type] = d[model][im_type] + [
                        Resize((352, 704))(
                            torch.unsqueeze(
                                torch.tensor(np.load(os.path.join(path_to_pics, el))), dim=0
                            )
                        ).numpy(force=True)
                        / 255.0
                    ]
            else:
                d[model][im_type] = d[model][im_type] + [
                    Resize((352, 704))(
                        torch.unsqueeze(
                            torch.tensor(np.load(os.path.join(path_to_pics, el))), dim=0
                        )
                    )
                    .squeeze()
                    .numpy(force=True)
                ]

    print("d: ", d)
    pprint(d)
    return d


def main(picture_folders: List[str]) -> None:
    """Setups a viser server (locally) running 3D simulations for viewing (erroneous) predictions."""
    server = viser.ViserServer()
    depths_dict = img_dicts(picture_folders=picture_folders)
    print(depths_dict)
    # Add some common GUI elements: number inputs, sliders, vectors, checkboxes.
    with server.add_gui_folder("Read-only"):
        gui_counter = server.add_gui_number(
            "Counter",
            initial_value=0,
            disabled=True,
        )

    with server.add_gui_folder("Editable"):
        gui_vector2 = server.add_gui_vector2(
            "Position",
            initial_value=(200.0, 200.0),
            step=5,
        )
        gui_vector3 = server.add_gui_vector3(
            "Pixeldensity",
            initial_value=(-1.0, -1.0, 1.0),  # (40.0, 40.0, 1.0)
            step=10,
        )
        campos = server.add_gui_vector3(
            "Campos",
            initial_value=(0, 0, 0),  # (37.0, 39.0, -6.0),
            step=0.1,
        )
        with server.add_gui_folder("Text toggle"):
            img_type = server.add_gui_dropdown(
                "Img type plane", ("preds", "img", "depth", "ScaledUncertainty", "std_dev")
            )
            model = server.add_gui_dropdown("model", tuple(depths_dict.keys()))
            img_num = server.add_gui_vector2(
                "img number",
                initial_value=(0, -1),
                step=1,
            )
            gui_rgb = server.add_gui_rgb(
                "Color",
                initial_value=(255, 0, 0),
            )

    depth = depths_dict[model.value][img_type.value][int(img_num.value[0])]
    monodepth_intrinsics = np.array(
        [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    monodepth_intrinsics[0, :] *= depth.shape[1]
    monodepth_intrinsics[1, :] *= depth.shape[0]
    monodepth_intrinsics = torch.tensor(monodepth_intrinsics, dtype=torch.float32)
    print(monodepth_intrinsics)

    K = torch.unsqueeze(
        monodepth_intrinsics, dim=0
    )  # get_intrinsics(depth.shape[0], depth.shape[1]), dim=0)
    E = torch.unsqueeze(torch.eye(4, dtype=torch.float32), dim=0)
    print(E.shape)
    # E[:, 2, 3] = 2.0
    print(K.dtype)
    print(E)

    korniacamera = kornia.geometry.camera.PinholeCamera(
        K, E, torch.ones(1) * depth.shape[0], torch.ones(1) * depth.shape[0]
    )
    print(depth.shape)
    img = depths_dict[model.value]["img"][int(img_num.value[0])]
    print(img)
    print(img[:, :20, :20])
    # Pre-generate a point cloud to send.s
    # point_positions = np.repeat(np.expand_dims(depth,axis=2), 3, axis=2) #np.random.uniform(low=-1.0, high=1.0, size=(5000, 3))
    point_positions = np.zeros((depth.shape[0] * depth.shape[1], 3))
    print(point_positions.shape)
    colors = (
        np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
    )  # (-1, 3))
    # colors=np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1,3).astype(np.uint8)#(-1, 3))

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            point_positions[y * depth.shape[1] + x, 0] = x - depth.shape[1] / 2
            point_positions[y * depth.shape[1] + x, 1] = y - depth.shape[0] / 2
            point_positions[y * depth.shape[1] + x, 2] = depth[
                y, x
            ]  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
            colors[y * depth.shape[1] + x, :] = (
                img[:, y, x] * 256
            )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])
    point_positions[point_positions == 0] = None
    print(colors.shape)
    counter = 0

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")
        client.camera.position = (-2.21710369, 0.25143548, 2.37680543)  #
        # position: [] (-2.20064452,  0.38278909,  0.10276915)  # (37, 39, -6)wxyz: [ 0.00224802  0.01156853  0.19074115 -0.98156963]
        # position: [0.98447588, 1.48619514, 0.49249996]
        client.camera.wxyz = (-0.00678133, 0.01222598, 0.06174117, -0.99799427)

        # client.camera.wxyz = (0,0,0,0)
        # This will run whenever we get a new camera!
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            print(f"New camera on client {client.client_id}!")
            print(f"\twxyz: {client.camera.wxyz}")
            print(f"\tposition: {client.camera.position}")

        # We can set the value of an input to a particular value. Changes are
        # Show the client ID in the GUI.
        gui_info = client.add_gui_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
        print("HI")

    while True:
        depth = depths_dict[model.value][img_type.value][int(img_num.value[0])]
        if img_type.value == "preds":
            depth = depths_dict[model.value][img_type.value][int(img_num.value[0])]
            depthcolumn = torch.zeros((depth.shape[0] * depth.shape[1]))
            img = depths_dict[model.value]["img"][int(img_num.value[0])]
            xycolumn = torch.zeros((depth.shape[0] * depth.shape[1], 2))
            point_positions = np.zeros((depth.shape[0] * depth.shape[1], 3))
            colors = (
                np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
            )  # (-1, 3))
            for y in range(depth.shape[0]):
                for x in range(depth.shape[1]):
                    xycolumn[y * depth.shape[1] + x, :] = torch.tensor([x, y])
                    depthcolumn[y * depth.shape[1] + x] = torch.tensor(
                        depth[y, x], dtype=torch.float32
                    )

                    colors[y * depth.shape[1] + x, :] = img[:, y, x] * 255

                    # if (y<5 or y>200) and x<5:
                    #    print(torch.squeeze(korniacamera.unproject(torch.tensor([[x,y]],dtype=torch.float32),depth[y,x])))
                    # point_positions[y * depth.shape[1] + x, :] = torch.squeeze(korniacamera.unproject(torch.tensor([[x,y]],dtype=torch.float32),depth[y,x]))
            xycolumn.to(torch.float32)
            depthcolumn = torch.unsqueeze(depthcolumn.to(torch.float32), dim=1)

            point_positions = torch.squeeze(korniacamera.unproject(xycolumn, depthcolumn)).numpy(
                force=True
            )
            point_positions[point_positions == 0] = None
        elif img_type.value in ["std_dev", "ScaledUncertainty"]:
            img = depth.squeeze()
            print("imgshape", img.shape)
            depth = depths_dict[model.value]["preds"][int(img_num.value[0])]
            depthcolumn = torch.zeros((depth.shape[0] * depth.shape[1]))
            xycolumn = torch.zeros((depth.shape[0] * depth.shape[1], 2))

            print(depth.shape)
            point_positions = np.zeros((depth.shape[0] * depth.shape[1], 3))
            colors = (
                np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
            )  # (-1, 3))
            for y in range(depth.shape[0]):
                for x in range(depth.shape[1]):
                    xycolumn[y * depth.shape[1] + x, :] = torch.tensor([x, y])
                    depthcolumn[y * depth.shape[1] + x] = torch.tensor(
                        depth[y, x], dtype=torch.float32
                    )

                    colors[y * depth.shape[1] + x, :] = img[:, y, x] * 255

                    # if (y<5 or y>200) and x<5:
                    #    print(torch.squeeze(korniacamera.unproject(torch.tensor([[x,y]],dtype=torch.float32),depth[y,x])))
                    # point_positions[y * depth.shape[1] + x, :] = torch.squeeze(korniacamera.unproject(torch.tensor([[x,y]],dtype=torch.float32),depth[y,x]))
            xycolumn.to(torch.float32)
            depthcolumn = torch.unsqueeze(depthcolumn.to(torch.float32), dim=1)

            point_positions = torch.squeeze(korniacamera.unproject(xycolumn, depthcolumn)).numpy(
                force=True
            )
            point_positions[point_positions == 0] = None

        else:
            point_positions = np.zeros((depth.shape[1] * depth.shape[2], 3))
            colors = (
                np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
            )  # (-1, 3))

            for y in range(depth.shape[0]):
                for x in range(depth.shape[1]):
                    point_positions[y * depth.shape[1] + x, 0] = x - depth.shape[1] / 2
                    point_positions[y * depth.shape[1] + x, 1] = y - depth.shape[0] / 2
                    point_positions[y * depth.shape[1] + x, 2] = depth[
                        y, x
                    ]  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
                    colors[y * depth.shape[1] + x, :] = (
                        img[:, y, x] * 255
                    )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])

            """ for y in range(depth.shape[1]):
                for x in range(depth.shape[2]):
                    print(depth.shape)
                    point_positions[y * depth.shape[2] + x, 0] = -1 * (x - depth.shape[2] / 2)
                    point_positions[y * depth.shape[2] + x, 1] = y - depth.shape[1] / 2
                    point_positions[
                        y * depth.shape[2] + x, 2
                    ] = 0  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
                    colors[y * depth.shape[2] + x, :] = (
                        img[:, y, x] * 255
                    )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])
 """
        clients = server.get_clients()
        # We can set the value of an input to a particular value. Changes are
        # automatically reflected in connected clients.
        gui_counter.value = counter

        # We can set the position of a scene node with `.position`, and read the value
        # of a gui element with `.value`. Changes are automatically reflected in
        # connected clients.
        server.add_point_cloud(
            "/point_cloud",
            points=point_positions
            / np.array(
                gui_vector3.value, dtype=np.float32
            ),  # * np.array(gui_vector3.value, dtype=np.float32),
            colors=colors,
            # np.tile(gui_rgb.value, point_positions.shape[0]).reshape((-1, 3))
            # * color_coeffs[:, None]
            position=gui_vector3.value + (0,),
        )
        if len(clients) > 0 and counter % 100 == 5:
            for k in clients.keys():
                clients[k].camera.position = (
                    -2.21710369,
                    0.25143548,
                    2.37680543,
                )  # (37, 39, -6)wxyz: [ 0.00224802  0.01156853  0.19074115 -0.98156963]
        # campos.value

        # We can use `.visible` and `.disabled` to toggle GUI elements.

        counter += 1
        time.sleep(0.02)


if __name__ == "__main__":
    main(
        [
            "2024_01_25_16_46_47_8006_Posthoc_Laplace",
            "2024_01_25_14_58_58_8006_Online_Laplace",
            "2024_01_26_18_56_18_8503_Ensemble",
            "2024_01_26_19_38_19_8508_Dropout",
        ]
    )
