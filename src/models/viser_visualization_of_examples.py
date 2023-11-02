"""GUI basics

Examples of basic GUI elements that we can create, read from, and write to."""

import time

import numpy as np

import viser
import os

depth = np.load("/home/jbv415/UncertainDepths/src/models/outputs/np_depth_2.npy")

def img_dicts():
    d = {"depth": [], "img": [], "preds": []}
    path_to_pics = "/home/jbv415/UncertainDepths/src/models/outputs/"
    for el in os.listdir("/home/jbv415/UncertainDepths/src/models/outputs/"):
        if el[-4:]!=".npy":
            continue
        d[el.split("_")[1]] = d[el.split("_")[1]] + [np.load(os.path.join(path_to_pics, el))]
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

        gui_slider = server.add_gui_slider(
            "Slider",
            min=0,
            max=100,
            step=1,
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
            initial_value=(0.0, 0.0, 0.0),
            step=1,
        )
        with server.add_gui_folder("Text toggle"):
            gui_checkbox_hide = server.add_gui_checkbox(
                "Hide",
                initial_value=False,
            )
            img_type = server.add_gui_text(
                "img type",
                initial_value="depth",
            )
            img_num = server.add_gui_vector2(
            "img number",
            initial_value=(2,-1),
            step=1,
        )   
            gui_button = server.add_gui_button("Button")
            gui_checkbox_disable = server.add_gui_checkbox(
                "Disable",
                initial_value=False,
            )
            gui_rgb = server.add_gui_rgb(
                "Color",
                initial_value=(255,0, 0),
            )
    depth = depths[img_type.value][int(img_num.value[0])]    
    print(depth.shape)
    # Pre-generate a point cloud to send.
    #point_positions = np.repeat(np.expand_dims(depth,axis=2), 3, axis=2) #np.random.uniform(low=-1.0, high=1.0, size=(5000, 3))
    point_positions = np.zeros((depth.shape[0]*depth.shape[1],3))
    print(point_positions.shape)
    colors=np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1,3).astype(np.uint8)#(-1, 3))
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            point_positions[y*depth.shape[1]+x,0] = (x - depth.shape[1]/2 )
            point_positions[y*depth.shape[1]+x,1] = (-y + depth.shape[0]/2 )
            point_positions[y*depth.shape[1]+x,2] = depth[y,x]    #color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
            colors[y*depth.shape[1]+x,:] = ((80-depth[y,x])/80)**2*np.array([255,0,0])
    point_positions[point_positions==0] = None
    print(colors.shape)
    counter = 0


    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        
        print("new client!")
        client.camera.position = (37,39,-26)
        client.camera.wxyz = (0,  9.80595212e-09,  4.99903834e-07, -9.99807668e-01)
        
        #client.camera.wxyz = (0,0,0,0)
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
        depth = depths[img_type.value][int(img_num.value[0])]    
        point_positions = np.zeros((depth.shape[0]*depth.shape[1],3))
        colors=np.tile(gui_rgb.value, point_positions.shape[0]).reshape(-1,3).astype(np.uint8)#(-1, 3))
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                point_positions[y*depth.shape[1]+x,0] = (x - depth.shape[1]/2 )
                point_positions[y*depth.shape[1]+x,1] = (-y + depth.shape[0]/2 )
                point_positions[y*depth.shape[1]+x,2] = depth[y,x]    #color_coeffs = np.random.uniform(0.4, 1.0, size=(point_positions.shape[0]))
                colors[y*depth.shape[1]+x,:] = ((80-depth[y,x])/80)**2*np.array([255,0,0])
        point_positions[point_positions==0] = None
    
        clients = server.get_clients()
        # We can set the value of an input to a particular value. Changes are
        # automatically reflected in connected clients.
        gui_counter.value = counter
        gui_slider.value = counter % 100

        # We can set the position of a scene node with `.position`, and read the value
        # of a gui element with `.value`. Changes are automatically reflected in
        # connected clients.
        server.add_point_cloud(
            "/point_cloud",
            points=point_positions/np.array(gui_vector3.value, dtype=np.float32), #* np.array(gui_vector3.value, dtype=np.float32),
            colors=colors,
                #np.tile(gui_rgb.value, point_positions.shape[0]).reshape((-1, 3))
                #* color_coeffs[:, None]
            
            position=gui_vector3.value + (0,),
        )

        # We can use `.visible` and `.disabled` to toggle GUI elements.
        gui_button.visible = not gui_checkbox_hide.value
        gui_rgb.disabled = gui_checkbox_disable.value

        counter += 1
        time.sleep(0.2)


if __name__ == "__main__":
    main()
