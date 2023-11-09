"""GUI basics

Examples of basic GUI elements that we can create, read from, and write to."""

import time
from copy import deepcopy
import numpy as np
import viser
import os
from torchvision.transforms import Resize
import torch


def img_dicts():
    d = {"depth": [], "img": [], "preds": []}
    d = {"Unet": deepcopy(d), "ZoeNK": deepcopy(d)}
    path_to_pics = "/home/jbv415/UncertainDepths/src/models/outputs/"
    visuals = os.listdir(path_to_pics)
    visuals.sort()
    for el in visuals:
        if el[-4:] != ".npy" or len(el.split("_")) != 4:
            continue
        print(el)
        model = el.split("_")[2]
        im_type = el.split("_")[1]
        print(el)
        print(torch.tensor(np.load(os.path.join(path_to_pics, el))).shape)
        if im_type == "img":  # Resize, so preds/depths can be color-corrected
            d[model][im_type] = d[model][im_type] + [
                Resize((352, 704))(torch.tensor(np.load(os.path.join(path_to_pics, el)))).numpy(
                    force=True
                )
            ]
        elif im_type == "depth":  # Resize, so preds/depths can be color-corrected
            d[model][im_type] = d[model][im_type] + [
                Resize((352, 704))(
                    torch.unsqueeze(torch.tensor(np.load(os.path.join(path_to_pics, el))), dim=0)
                )
                .squeeze()
                .numpy(force=True)
            ]
        else:
            d[model][im_type] = d[model][im_type] + [
                Resize((352, 704))(
                    torch.unsqueeze(torch.tensor(np.load(os.path.join(path_to_pics, el))), dim=0)
                )
                .squeeze()
                .numpy(force=True)
            ]
    return d


def main():
    server = viser.ViserServer()
    depths = img_dicts()
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
            "Position",
            initial_value=(40.0, 40.0, 1.0),
            step=10,
        )
        campos = server.add_gui_vector3(
            "Campos",
            initial_value=(37.0, 39.0, -6.0),
            step=0.1,
        )
        with server.add_gui_folder("Text toggle"):
            img_type = server.add_gui_dropdown("Img type plane", ("preds", "depth", "img"))
            model = server.add_gui_dropdown("model", ("ZoeNK", "Unet", "ZoeNK"))
            img_num = server.add_gui_vector2(
                "img number",
                initial_value=(2, -1),
                step=1,
            )
            gui_rgb = server.add_gui_rgb(
                "Color",
                initial_value=(255, 0, 0),
            )
    depth = depths[model.value][img_type.value][int(img_num.value[0])]
    print(depth.shape)
    img = depths[model.value]["img"][int(img_num.value[0])]
    print(img.shape)
    print(img)
    print(img[:, :20, :20])
    # Pre-generate a point cloud to send.
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
            point_positions[y * depth.shape[1] + x, 1] = -y + depth.shape[0] / 2
            point_positions[y * depth.shape[1] + x, 2] = depth[
                y, x
            ]  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
            colors[y * depth.shape[1] + x, :] = (
                img[:, y, x] * 255
            )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])
    point_positions[point_positions == 0] = None
    print(colors.shape)
    counter = 0

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")
        client.camera.position = (37, 39, -6)
        client.camera.wxyz = (0, 9.80595212e-09, 4.99903834e-07, -9.99807668e-01)

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

    while True:
        depth = depths[model.value][img_type.value][int(img_num.value[0])]
        img = depths[model.value]["img"][int(img_num.value[0])]

        print(depth.shape)

        if img_type.value != "img":
            point_positions = np.zeros((depth.shape[0] * depth.shape[1], 3))
            colors = (
                np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
            )  # (-1, 3))
            for y in range(depth.shape[0]):
                for x in range(depth.shape[1]):
                    point_positions[y * depth.shape[1] + x, 0] = -1 * (x - depth.shape[1] / 2)
                    point_positions[y * depth.shape[1] + x, 1] = -y + depth.shape[0] / 2
                    point_positions[y * depth.shape[1] + x, 2] = depth[
                        y, x
                    ]  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
                    colors[y * depth.shape[1] + x, :] = (
                        img[:, y, x] * 255
                    )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])
            point_positions[point_positions == 0] = None
        else:
            point_positions = np.zeros((depth.shape[1] * depth.shape[2], 3))
            colors = (
                np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1, 3).astype(np.uint8)
            )  # (-1, 3))
            for y in range(depth.shape[1]):
                for x in range(depth.shape[2]):
                    print(depth.shape)
                    point_positions[y * depth.shape[2] + x, 0] = -1 * (x - depth.shape[2] / 2)
                    point_positions[y * depth.shape[2] + x, 1] = -y + depth.shape[1] / 2
                    point_positions[
                        y * depth.shape[2] + x, 2
                    ] = 0  # color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
                    colors[y * depth.shape[2] + x, :] = (
                        img[:, y, x] * 255
                    )  # ((80-depth[y,x])/80)**2*np.array([255,0,0])

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
                clients[k].camera.position = campos.value

        # We can use `.visible` and `.disabled` to toggle GUI elements.

        counter += 1
        time.sleep(0.02)


if __name__ == "__main__":
    main()
