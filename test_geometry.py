import numpy as np
import open3d as o3
import slam_lib.geometry


def main():
    # bunny_path = './data/bunny/reconstruction/bun_zipper.ply'
    # bunny = o3.geometry.PointCloud()
    # bunny = o3.io.read_point_cloud(bunny_path)
    # bunny = bunny.voxel_down_sample(0.002)

    wall = o3.geometry.PointCloud()
    xs, ys = np.arange(-50, 50), np.arange(-50, 50)
    xs, ys = np.meshgrid(xs, ys)
    pts = np.asarray([xs, ys, np.zeros(xs.shape) + 50])
    pts = pts.transpose((1, 2, 0)).reshape(-1, 3)

    wall.points = o3.utility.Vector3dVector(pts)

    lines = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 0]])
    
    index_lines_2_pts, lines = slam_lib.geometry.nearest_points_2_lines(lines, pts)

    print(lines)
    print(index_lines_2_pts)

    for i, index_line_2_pt in enumerate(index_lines_2_pts):
        print(lines[i], pts[index_line_2_pt])

    pts = pts[index_lines_2_pts]
    lines_proj = slam_lib.geometry.closet_vector_2_point(lines, pts)

    print('lines normalized', lines)
    print('closet pts ', pts)
    print('line proj', lines_proj)

    mesh = o3.geometry.TriangleMesh()
    frame = mesh.create_coordinate_frame(size=10)
    spheres = []
    for i, pt in enumerate(pts):
        print(pt)
        tf = np.eye(4)
        tf[:3, -1] = pt
        spheres.append(mesh.create_sphere(radius=0.5))
        spheres[-1].transform(tf)
    o3.visualization.draw_geometries([wall, frame, spheres[0]])
    o3.visualization.draw_geometries([wall, frame, spheres[1]])
    o3.visualization.draw_geometries([wall, frame, spheres[2]])


if __name__ == '__main__':
    main()
