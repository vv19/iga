import pyvista as pv
import numpy as np
import torch
from iga.utils.common_utils import transform_pcd


class Visualiser:

    def __init__(self, off_screen=False, save_dir=None, config=None):
        self.save_dir = save_dir
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=(1856, 1024), title='Implicit Graph Alignment')
        self.off_screen = off_screen
        self.plotter.background_color = 'white'
        self.shown = None
        self.config = config

    def add_pcd_to_plotter(self, pcd, color='r', radius=0.015, name=None, opacity=1., geom_type='sphere',
                           cmap='Paired'):

        if geom_type.lower() == 'triangle':
            geom = pv.Cone(radius=2 * radius, resolution=3, direction=[0., 0., 1.], height=2 * radius)
        elif geom_type.lower() == 'cube':
            geom = pv.Box(bounds=[-radius, radius, -radius, radius, -radius, radius])
        else:
            geom = pv.Sphere(radius=radius)

        points = pv.PolyData(pcd)
        if color == 'rainbow':
            points['point_color'] = np.arange(len(pcd))
            return self.plotter.add_mesh(points.glyph(geom=geom, scale=False),
                                         scalars='point_color',
                                         cmap=cmap,
                                         show_scalar_bar=False,
                                         name=name)
        else:
            return self.plotter.add_mesh(points.glyph(geom=geom), color=color, name=name, opacity=opacity)

    def draw_coord_frame(self, T, name=None, opacity=1., scale=1.):
        tip_length = 0.25  # * scale
        tip_radius = 0.08 * scale
        tip_resolution = 20
        shaft_radius = 0.04 * scale
        shaft_resolution = 20
        scale = 0.05  # * scale
        if name is None:
            name = str(np.random.uniform())
        self.plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 0],
                                       tip_length=tip_length,
                                       tip_radius=tip_radius,
                                       tip_resolution=tip_resolution,
                                       shaft_radius=shaft_radius,
                                       shaft_resolution=shaft_resolution,
                                       scale=scale), color='r', name=name + 'x', opacity=opacity)
        self.plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 1],
                                       tip_length=tip_length,
                                       tip_radius=tip_radius,
                                       tip_resolution=tip_resolution,
                                       shaft_radius=shaft_radius,
                                       shaft_resolution=shaft_resolution,
                                       scale=scale), color='g', name=name + 'y', opacity=opacity)
        self.plotter.add_mesh(pv.Arrow(T[:3, 3], T[:3, 2],
                                       tip_length=tip_length,
                                       tip_radius=tip_radius,
                                       tip_resolution=tip_resolution,
                                       shaft_radius=shaft_radius,
                                       shaft_resolution=shaft_resolution,
                                       scale=scale), color='b', name=name + 'z', opacity=opacity)

    def visualise_graph(self, hetero_data, label_mask):
        #########################################################################################################
        if self.config is None:
            scale = 1.
        else:
            scale = self.config['pcd_scaling']
        radius_node = 0.01 * scale
        radius_energy = 0.03 * scale
        batch_spacings = 2. * scale
        offset_spacings = 0.7 * scale
        energy_lift = 0.3 * scale
        #########################################################################################################
        # Visualise the graph
        # self.clear()
        labels = hetero_data['e'].label
        pos_np = hetero_data['local'].pos.detach().cpu().numpy()
        pos_np_a = np.zeros((len(pos_np[(hetero_data['local'].a_b_mask == 1).cpu().numpy()]), 3))
        pos_np_b = np.zeros((len(pos_np[(hetero_data['local'].a_b_mask == 0).cpu().numpy()]), 3))
        pos_e_np = np.zeros((hetero_data['e'].num_nodes, 3))

        offset = np.array([0., 0, 0])
        k = 0
        last_a = 0
        last_b = 0
        for i in range(len(label_mask)):

            if label_mask[i] == 0:
                offset[1] += offset_spacings
                offset[2] = 0
            elif label_mask[i] == -1:
                offset[2] += offset_spacings
            else:
                offset[0] += offset_spacings
                offset[1] = 0
                offset[2] = 0

            pos_np_a_local = pos_np[torch.logical_and(hetero_data['local'].a_b_mask == 1,
                                                      hetero_data['local'].batch_grouped == i).cpu().numpy()]
            pos_np_b_local = pos_np[torch.logical_and(hetero_data['local'].a_b_mask == 0,
                                                      hetero_data['local'].batch_grouped == i).cpu().numpy()]

            pos_np[(hetero_data['local'].batch_grouped == i).cpu().numpy()] += - pos_np_a_local.mean(axis=0) + offset

            pos_np_a[last_a:last_a + len(pos_np_a_local)] = pos_np_a_local - pos_np_a_local.mean(axis=0) + offset
            pos_np_b[last_b:last_b + len(pos_np_b_local)] = pos_np_b_local - pos_np_a_local.mean(axis=0) + offset

            if label_mask[i] != -1:
                pos_e_np[k, :] = np.mean(pos_np_a[last_a:last_a + len(pos_np_a_local)], axis=0) + np.array(
                    [0, 0., energy_lift])
                k += 1

            last_a += len(pos_np_a_local)
            last_b += len(pos_np_b_local)

        for i in range(torch.max(hetero_data['local'].true_batch).item() + 1):
            pos_np[(hetero_data['local'].true_batch == i).cpu().numpy()] += np.array([batch_spacings * i, 0, 0])
            pos_e_np[(hetero_data['e'].e_batch_demo == i).cpu().numpy()] += np.array([batch_spacings * i, 0, 0])

        pos_np_a = pos_np[(hetero_data['local'].a_b_mask == 1).cpu().numpy()]
        pos_np_b = pos_np[(hetero_data['local'].a_b_mask == 0).cpu().numpy()]
        pos_np = {'local': pos_np, 'e': pos_e_np}
        pos_e_positives = pos_e_np[labels.cpu().numpy() == 1, :]

        self.add_pcd_to_plotter(pos_np_a, color='r', radius=radius_node, name=f'a', opacity=1, geom_type='triangle')
        self.add_pcd_to_plotter(pos_np_b, color='g', radius=radius_node, name=f'b', opacity=1, geom_type='cube')
        if 'e' in hetero_data.node_types:
            self.add_pcd_to_plotter(pos_e_np, color='b', radius=radius_energy, name=f'e', opacity=1, geom_type='sphere')
            self.add_pcd_to_plotter(pos_e_positives, color='r', radius=radius_energy, name=f'e_pos', opacity=1,
                                    geom_type='sphere')
        self.plotter.add_text(f'Only the Nodes', font_size=20, position='upper_left', name='text')
        if self.shown is None:
            self.plotter.show(auto_close=False)
            self.shown = True
        edge_colours = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w', 'r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
        for k, edge_info in enumerate(hetero_data.edge_types):
            src, edge_type, dst = edge_info
            self.plotter.add_text(f'{src} - {edge_type} - {dst}', font_size=20, position='upper_left', name='text')
            line_start = pos_np[src][hetero_data[src, edge_type, dst].edge_index[0, :].cpu().numpy()]
            line_end = pos_np[dst][hetero_data[src, edge_type, dst].edge_index[1, :].cpu().numpy()]

            self.add_pcd_to_plotter(line_start, color='m', radius=0.05, name=f'start', opacity=0.5,
                                    geom_type='sphere')

            for j in range(len(line_start)):
                self.plotter.add_mesh(pv.Line(line_start[j], line_end[j]), line_width=2, color=edge_colours[k],
                                      opacity=1)

            self.plotter.show(auto_close=False)

    def clear(self):
        self.plotter.clear()
        self.plotter.enable_lightkit()

    @torch.no_grad()
    def visualise_energy(self, forward, batch, resolution_xy=(50, 50), size_xy=(1.5, 1.5), device='cuda:0',
                         override_meshes=False, global_step=None):
        if self.config is None:
            scale = 1.
        else:
            scale = self.config['pcd_scaling']
        if override_meshes:
            self.clear()
            names = ['energy_landscape', 'a', 'b', 'minimum', 'minimum_gt', 'a_c', 'b_c']
        else:
            names = [None] * 7

        plane_mesh = pv.Plane(i_size=size_xy[0], j_size=size_xy[1], i_resolution=resolution_xy[0],
                              j_resolution=resolution_xy[1])
        offsets = torch.tensor(plane_mesh.points).to(device) - torch.mean(batch.pos_b_0, dim=0)
        T_noise = torch.eye(4, device=device).repeat(offsets.shape[0] + 1, 1, 1)
        T_noise[1:, :2, 3] = offsets[:, :2]

        logits, y = forward(batch, T_noise, num_negatives=len(offsets))
        energy_landscape = logits.cpu().float().numpy()[0, -len(offsets):]

        pos_a_np = batch.pos_a_0.cpu().numpy()
        pos_b_np = batch.pos_b_0.cpu().numpy()

        self.visualised_energy_landscape = plane_mesh.points
        self.visualised_plane_mesh = plane_mesh
        # Normalise energy landscape
        energy_landscape = (energy_landscape - np.min(energy_landscape) + 1e-6) / (
                np.max(energy_landscape) - np.min(energy_landscape) + 1e-6)

        plane_mesh['energy'] = energy_landscape
        plane_mesh.set_active_scalars('energy')

        # 2D projection of the energy landscape. z value is constant.
        plane_mesh_projection = plane_mesh.copy()
        plane_mesh_projection.points[:, 2] = scale * 0.1 + np.mean(pos_b_np[:, 2], axis=0)

        plane_mesh.points[:, 2] = energy_landscape * scale + 0.4 * scale + np.mean(pos_b_np[:, 2], axis=0)
        self.mean_b_pos = torch.mean(batch.pos_b_1, dim=0).cpu().numpy()

        self.plotter.add_mesh(plane_mesh, scalars='energy', show_edges=False, opacity=0.7, name=names[0],
                              show_scalar_bar=False)
        self.plotter.add_mesh(plane_mesh_projection, scalars='energy', show_edges=False, opacity=0.7,
                              show_scalar_bar=False)

        self.add_pcd_to_plotter(pos_a_np, color='r', radius=0.005 * scale,
                                name=names[1],
                                opacity=1, geom_type='triangle')
        self.add_pcd_to_plotter(pos_b_np, color='g', radius=0.005 * scale,
                                name=names[2],
                                opacity=1, geom_type='cube')

        self.add_pcd_to_plotter(np.mean(pos_b_np, axis=0) + np.array([0, 0, 0.1 * scale]),
                                color='m', radius=0.01 * scale,
                                name=names[3],
                                opacity=1)

        self.add_pcd_to_plotter(plane_mesh.points[np.argmin(energy_landscape)],
                                color='y', radius=0.01 * scale,
                                name=names[4],
                                opacity=1)

        if global_step is not None:
            self.plotter.add_text(f'Training Step: {global_step}', font_size=20, position='upper_left', name='text')
        if self.save_dir is None or not self.config['record']:
            self.plotter.show(auto_close=False)
        else:
            if global_step is not None:
                self.plotter.show(screenshot=f'{self.save_dir}/log_image_{global_step}.png', auto_close=False)
            else:
                self.plotter.show(screenshot=f'{self.save_dir}/log_image.png', auto_close=False)

    def visualise_trajectory(self, data, traj, graph_means, context_length=5, overlayed=False, graph=False):
        data = data.clone()
        self.clear()
        # Add "Conditional" points to the plotter.
        if self.config is None:
            scale = 1.
        else:
            scale = self.config['pcd_scaling']
        colours = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b', 'r', 'g', 'y', 'm', 'c', 'k', 'w', 'b', 'r', 'g', 'y', 'm']

        shuffled_demo_indices = np.random.permutation(list(range(1, context_length)))

        if not overlayed:
            for k, i in enumerate(shuffled_demo_indices):
                data[f'pos_b_{i}'] += torch.tensor([0.3, -0.8 + (k + 1) * 0.3, 0.2],
                                                   device=data[f'pos_b_{i}'].device) * scale
                data[f'pos_a_{i}'] += torch.tensor([0.3, -0.8 + (k + 1) * 0.3, 0.2],
                                                   device=data[f'pos_a_{i}'].device) * scale

                data[f'centres_b_{i}'] += torch.tensor([0.3, -0.8 + (k + 1) * 0.3, 0.2],
                                                       device=data[f'centres_b_{i}'].device) * scale
                data[f'centres_a_{i}'] += torch.tensor([0.3, -0.8 + (k + 1) * 0.3, 0.2],
                                                       device=data[f'centres_a_{i}'].device) * scale

        ################################################################################################################
        cmaps_a = ['Pastel1', 'Paired', 'Accent']
        cmaps_b = ['Pastel2', 'Set1', 'Set2']
        ################################################################################################################
        b_pos_c_demos = []
        a_pos_c_demos = []
        for k, i in enumerate(shuffled_demo_indices):
            if graph:
                b_pos_c = data[f'centres_b_{i}'].detach().cpu().numpy()
                b_pos_c_demos.append(b_pos_c)
                a_pos_c = data[f'centres_a_{i}'].detach().cpu().numpy()
                a_pos_c_demos.append(a_pos_c)
                self.add_graph(a_pos_c, b_pos_c, cmap_a=cmaps_a[0], cmap_b=cmaps_b[0], scale=scale,
                               add_edges=True, name=f'demo_{k}')

            else:
                b_pos = data[f'pos_b_{i}'].detach().cpu().numpy()
                a_pos = data[f'pos_a_{i}'].detach().cpu().numpy()

                self.add_pcd_to_plotter(b_pos, color=colours[k], name=f'b_pos_{i}',
                                        opacity=1. if not overlayed else 0.5, radius=0.003 * scale)
                self.add_pcd_to_plotter(a_pos, color=colours[k], name=f'a_pos_{i}',
                                        opacity=1. if not overlayed else 0.5, radius=0.003 * scale)

        # That is the point cloud we'll move according to the trajectory.
        b_pos = data.pos_b_0.detach().cpu().numpy()
        a_pos = data.pos_a_0.detach().cpu().numpy()

        b_pos_c = data[f'centres_b_0'].detach().cpu().numpy()
        a_pos_c = data[f'centres_a_0'].detach().cpu().numpy()

        # Visualising the trajectory.
        shown = False
        self.plotter.camera_position = [(-82.60745973333616, -67.00533027924799, 122.97963194041621),
                                        (19.004851458520573, -9.481480986835706, 2.437432325276669),
                                        (0.7504551417407374, 0.05460458033914778, 0.6586618404317011)]
        if not graph:
            self.add_pcd_to_plotter(a_pos, color='b', name='a_pos_new', opacity=1., radius=0.003 * scale)
        for i in range(len(traj)):
            b_pos -= graph_means[i].detach().cpu().numpy()
            b_pos = transform_pcd(b_pos, traj[i].squeeze().detach().cpu().numpy())
            b_pos += graph_means[i].detach().cpu().numpy()

            b_pos_c -= graph_means[i].detach().cpu().numpy()
            b_pos_c = transform_pcd(b_pos_c, traj[i].squeeze().detach().cpu().numpy())
            b_pos_c += graph_means[i].detach().cpu().numpy()

            if graph:
                self.add_graph(a_pos_c, b_pos_c, cmap_a=cmaps_a[0], cmap_b=cmaps_b[0], scale=scale,
                               add_edges=True, name='test')

                for j in range(len(a_pos_c_demos)):
                    if not shown:
                        connect_nodes(self.plotter, a_pos_c_demos[j], a_pos_c, colour='gray', line_width=1, opacity=0.5,
                                      name=f'cond_a_{j}')
                    connect_nodes(self.plotter, b_pos_c_demos[j], b_pos_c, colour='gray', line_width=1, opacity=0.5,
                                  name=f'cond_b_{j}')
            else:
                # self.add_pcd_to_plotter(b_pos, color='b', name='b_pos_new', opacity=1., radius=0.005 * scale)
                # For speed.
                self.plotter.add_mesh(pv.PolyData(b_pos), render_points_as_spheres=True, color='b', name='b_pos_new',
                                      point_size=12)

            if not shown:
                self.plotter.add_text(f'Optimisation Starting Point, press q to continue',
                                      font_size=20, position='upper_left', name='text', color='k')
                self.plotter.show(auto_close=False)
                shown = True
                continue
            self.plotter.add_text(f'Optimisation Step {i + 1}',
                                  font_size=20, position='upper_left', name='text', color='k')
            self.add_pcd_to_plotter(graph_means[i].detach().cpu().numpy(), color='y', opacity=1., radius=0.005 * scale)

        self.plotter.add_text(f'Optimisation End Point, press q to continue',
                              font_size=20, position='upper_left', name='text', color='k')
        self.plotter.show(auto_close=False)

    def add_graph(self, pcd_1, pcd_2, cmap_a='Pastel1', cmap_b='Pastel2', scale=1., add_edges=True, name=None):
        color_light_blue = '#A3C1DA'
        color_dark_green = '#006400'
        color_dark_yellow = '#FFCC00'
        color_gray = '#808080'
        color_dark_blue = '#00008B'

        self.add_pcd_to_plotter(pcd_1, color='rainbow', radius=0.01 * scale, opacity=1., geom_type='sphere',
                                cmap=cmap_a,
                                name=name + '_a')
        self.add_pcd_to_plotter(pcd_2, color='rainbow', radius=0.01 * scale, opacity=1., geom_type='cube', cmap=cmap_b,
                                name=name + '_b')
        if add_edges:
            connect_nodes(self.plotter, pcd_1, pcd_2, colour=color_dark_green, line_width=2, name=name + '_ab')
            connect_nodes(self.plotter, pcd_1, pcd_1, colour=color_dark_yellow, line_width=2, name=name + '_aa')
            connect_nodes(self.plotter, pcd_2, pcd_2, colour=color_dark_yellow, line_width=2, name=name + '_bb')


def add_parts_to_potter(p, pcd, centre_idx, cmap='Paired', opacity=1., scale=1., name=None):
    geom_pcd = pv.Sphere(radius=0.003 * scale)
    pcd_a = pv.PolyData(pcd)
    pcd_a['point_color'] = centre_idx[np.arange(len(pcd))]
    p.add_mesh(pcd_a.glyph(geom=geom_pcd, scale=False), scalars='point_color',
               cmap=cmap,
               show_scalar_bar=False, opacity=opacity, name=name)


def connect_nodes(p, pcd_1, pcd_2, colour='r', line_width=3, opacity=1., name=None):
    # Connect all the nodes in pcd_1 with all the nodes in pcd_2
    edge_idx = torch.cartesian_prod(
        torch.arange(len(pcd_1), dtype=torch.int64),
        torch.arange(len(pcd_1), dtype=torch.int64)).contiguous().t()

    line_start = pcd_1[edge_idx[0, :].cpu().numpy()]
    line_end = pcd_2[edge_idx[1, :].cpu().numpy()]

    vertices = np.concatenate([line_start, line_end], axis=0)
    lines = np.array([np.arange(len(line_start)), np.arange(len(line_start), len(vertices))]).T
    lines = np.concatenate([np.array([[2] * len(line_start)]).T, lines], axis=1)

    mesh = pv.PolyData(vertices, lines=lines)
    p.add_mesh(mesh, color=colour, line_width=line_width, opacity=opacity, name=name)
