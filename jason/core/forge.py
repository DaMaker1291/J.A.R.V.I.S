"""
J.A.S.O.N. Forge Protocol
Photogrammetry-to-STL Autonomous Pipeline
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# 3D processing libraries
try:
    import open3d as o3d
    open3d_available = True
except ImportError:
    open3d_available = False

try:
    import trimesh
    trimesh_available = True
except ImportError:
    trimesh_available = False

logger = logging.getLogger(__name__)

class ForgeManager:
    """Forge Protocol: Photogrammetry-to-STL autonomous pipeline"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Forge settings
        forge_config = self.config.get('forge', {})
        self.input_formats = forge_config.get('input_formats', ['ply', 'obj', 'stl', 'xyz', 'pcd'])
        self.output_format = forge_config.get('output_format', 'stl')
        self.simplify_ratio = forge_config.get('simplify_ratio', 0.5)  # Mesh simplification
        self.voxel_size = forge_config.get('voxel_size', 0.01)  # For point cloud processing
        self.auto_clean = forge_config.get('auto_clean', True)

        # Supported formats
        self.supported_input_formats = {
            'ply': self._load_ply,
            'obj': self._load_obj,
            'stl': self._load_stl,
            'xyz': self._load_xyz,
            'pcd': self._load_pcd
        }

        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'stl_generated': 0,
            'processing_errors': 0,
            'total_input_size_mb': 0.0,
            'total_output_size_mb': 0.0
        }

        # Check dependencies
        if not open3d_available and not trimesh_available:
            logger.warning("Neither Open3D nor trimesh available. Forge protocol limited.")

    def process_photogrammetry_file(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single photogrammetry file to STL"""
        input_path = Path(input_path)

        if not input_path.exists():
            return {'success': False, 'error': f'Input file does not exist: {input_path}'}

        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_forge.stl"
        else:
            output_path = Path(output_path)

        # Check input format
        ext = input_path.suffix.lower().lstrip('.')
        if ext not in self.supported_input_formats:
            return {'success': False, 'error': f'Unsupported format: {ext}'}

        try:
            logger.info(f"Processing photogrammetry file: {input_path}")

            # Load the file
            mesh = self.supported_input_formats[ext](input_path)

            if mesh is None:
                return {'success': False, 'error': 'Failed to load input file'}

            # Process the mesh
            processed_mesh = self._process_mesh(mesh)

            # Save as STL
            success = self._save_stl(processed_mesh, output_path)

            if success:
                # Update statistics
                input_size = input_path.stat().st_size / (1024 * 1024)
                output_size = output_path.stat().st_size / (1024 * 1024)

                self.stats['files_processed'] += 1
                self.stats['stl_generated'] += 1
                self.stats['total_input_size_mb'] += input_size
                self.stats['total_output_size_mb'] += output_size

                result = {
                    'success': True,
                    'input_file': str(input_path),
                    'output_file': str(output_path),
                    'input_size_mb': input_size,
                    'output_size_mb': output_size,
                    'compression_ratio': output_size / input_size if input_size > 0 else 0
                }

                logger.info(f"Successfully converted {input_path} to {output_path}")
                return result
            else:
                self.stats['processing_errors'] += 1
                return {'success': False, 'error': 'Failed to save STL file'}

        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error processing {input_path}: {e}")
            return {'success': False, 'error': str(e)}

    def batch_process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process all photogrammetry files in a directory"""
        input_dir = Path(input_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            return {'success': False, 'error': f'Input directory does not exist: {input_dir}'}

        if output_dir is None:
            output_dir = input_dir / "forge_output"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        results = {
            'success': True,
            'total_files': 0,
            'processed_files': 0,
            'errors': 0,
            'results': []
        }

        # Find all supported files
        for ext in self.input_formats:
            for input_file in input_dir.rglob(f"*.{ext}"):
                results['total_files'] += 1

                # Create output path
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / f"{relative_path.stem}_forge.stl"

                # Process the file
                result = self.process_photogrammetry_file(str(input_file), str(output_file))
                results['results'].append(result)

                if result['success']:
                    results['processed_files'] += 1
                else:
                    results['errors'] += 1

        logger.info(f"Batch processing completed: {results['processed_files']}/{results['total_files']} files processed")
        return results

    def _load_ply(self, filepath: Path) -> Optional[Any]:
        """Load PLY file"""
        if open3d_available:
            try:
                mesh = o3d.io.read_triangle_mesh(str(filepath))
                if mesh.is_empty():
                    # Try as point cloud
                    pcd = o3d.io.read_point_cloud(str(filepath))
                    if not pcd.is_empty():
                        # Convert point cloud to mesh
                        mesh = self._point_cloud_to_mesh(pcd)
                    else:
                        return None
                return mesh
            except Exception as e:
                logger.warning(f"Open3D PLY load failed: {e}")

        if trimesh_available:
            try:
                mesh = trimesh.load(str(filepath))
                return mesh
            except Exception as e:
                logger.warning(f"Trimesh PLY load failed: {e}")

        return None

    def _load_obj(self, filepath: Path) -> Optional[Any]:
        """Load OBJ file"""
        if trimesh_available:
            try:
                mesh = trimesh.load(str(filepath))
                return mesh
            except Exception as e:
                logger.warning(f"Trimesh OBJ load failed: {e}")

        if open3d_available:
            try:
                mesh = o3d.io.read_triangle_mesh(str(filepath))
                return mesh
            except Exception as e:
                logger.warning(f"Open3D OBJ load failed: {e}")

        return None

    def _load_stl(self, filepath: Path) -> Optional[Any]:
        """Load STL file"""
        if trimesh_available:
            try:
                mesh = trimesh.load(str(filepath))
                return mesh
            except Exception as e:
                logger.warning(f"Trimesh STL load failed: {e}")

        if open3d_available:
            try:
                mesh = o3d.io.read_triangle_mesh(str(filepath))
                return mesh
            except Exception as e:
                logger.warning(f"Open3D STL load failed: {e}")

        return None

    def _load_xyz(self, filepath: Path) -> Optional[Any]:
        """Load XYZ point cloud file"""
        if open3d_available:
            try:
                pcd = o3d.io.read_point_cloud(str(filepath))
                if not pcd.is_empty():
                    # Convert to mesh
                    return self._point_cloud_to_mesh(pcd)
            except Exception as e:
                logger.warning(f"Open3D XYZ load failed: {e}")

        # Fallback: manual parsing
        try:
            points = []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if points:
                points = np.array(points)
                if open3d_available:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    return self._point_cloud_to_mesh(pcd)
                elif trimesh_available:
                    # Create simple mesh from points
                    # This is a basic approximation
                    return self._points_to_simple_mesh(points)

        except Exception as e:
            logger.warning(f"Manual XYZ load failed: {e}")

        return None

    def _load_pcd(self, filepath: Path) -> Optional[Any]:
        """Load PCD point cloud file"""
        if open3d_available:
            try:
                pcd = o3d.io.read_point_cloud(str(filepath))
                if not pcd.is_empty():
                    return self._point_cloud_to_mesh(pcd)
            except Exception as e:
                logger.warning(f"Open3D PCD load failed: {e}")

        return None

    def _point_cloud_to_mesh(self, pcd) -> Optional[Any]:
        """Convert point cloud to mesh using Poisson reconstruction"""
        if not open3d_available:
            return None

        try:
            # Estimate normals
            pcd.estimate_normals()

            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )

            # Remove low-density vertices
            if densities:
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)

            return mesh

        except Exception as e:
            logger.warning(f"Point cloud to mesh conversion failed: {e}")
            return None

    def _points_to_simple_mesh(self, points: np.ndarray) -> Optional[Any]:
        """Create a simple mesh from points (basic triangulation)"""
        if not trimesh_available:
            return None

        try:
            # This is a very basic implementation
            # In practice, you'd want better triangulation
            from scipy.spatial import Delaunay

            # Project to 2D for triangulation (using XY plane)
            points_2d = points[:, :2]

            # Triangulate
            tri = Delaunay(points_2d)
            faces = tri.simplices

            # Create trimesh
            mesh = trimesh.Trimesh(vertices=points, faces=faces)
            return mesh

        except Exception as e:
            logger.warning(f"Simple mesh creation failed: {e}")
            return None

    def _process_mesh(self, mesh) -> Any:
        """Process and clean the mesh"""
        if mesh is None:
            return None

        try:
            # Determine mesh type and process accordingly
            if open3d_available and hasattr(mesh, 'compute_vertex_normals'):
                # Open3D mesh
                if self.auto_clean:
                    # Remove duplicated vertices
                    mesh.remove_duplicated_vertices()
                    # Remove degenerate triangles
                    mesh.remove_degenerate_triangles()

                # Simplify mesh if too complex
                if len(mesh.triangles) > 50000:
                    mesh = mesh.simplify_quadric_decimation(
                        target_number_of_triangles=int(len(mesh.triangles) * self.simplify_ratio)
                    )

                # Compute normals
                mesh.compute_vertex_normals()

                return mesh

            elif trimesh_available and hasattr(mesh, 'vertices'):
                # Trimesh mesh
                if self.auto_clean:
                    # Remove duplicate faces
                    mesh.remove_duplicate_faces()
                    # Remove degenerate faces
                    mesh.remove_degenerate_faces()

                # Simplify if too complex
                if len(mesh.faces) > 50000:
                    mesh = mesh.simplify_quadratic_decimation(
                        int(len(mesh.faces) * self.simplify_ratio)
                    )

                # Ensure mesh is watertight for STL
                if not mesh.is_watertight:
                    logger.warning("Mesh is not watertight - may have issues in 3D printing")

                return mesh

            else:
                return mesh

        except Exception as e:
            logger.warning(f"Mesh processing failed: {e}")
            return mesh

    def _save_stl(self, mesh, output_path: Path) -> bool:
        """Save mesh as STL file"""
        try:
            if open3d_available and hasattr(mesh, 'compute_vertex_normals'):
                # Open3D mesh
                return o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=True)

            elif trimesh_available and hasattr(mesh, 'vertices'):
                # Trimesh mesh
                mesh.export(str(output_path))
                return output_path.exists()

            else:
                logger.error("No suitable library available for STL export")
                return False

        except Exception as e:
            logger.error(f"STL export failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats"""
        return list(self.supported_input_formats.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'files_processed': 0,
            'stl_generated': 0,
            'processing_errors': 0,
            'total_input_size_mb': 0.0,
            'total_output_size_mb': 0.0
        }

    def validate_mesh_for_printing(self, mesh) -> Dict[str, Any]:
        """Validate mesh for 3D printing"""
        validation = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }

        try:
            if trimesh_available and hasattr(mesh, 'vertices'):
                # Check if manifold
                if not mesh.is_watertight:
                    validation['issues'].append('Mesh is not watertight')
                    validation['recommendations'].append('Repair holes in the mesh')

                # Check for self-intersections
                if mesh.self_intersecting():
                    validation['issues'].append('Mesh has self-intersections')
                    validation['recommendations'].append('Fix self-intersecting geometry')

                # Check volume
                if mesh.volume < 0:
                    validation['issues'].append('Mesh has negative volume (inverted normals)')
                    validation['recommendations'].append('Flip face normals')

                # Check bounds
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                max_dimension = max(size)
                if max_dimension > 200:  # mm
                    validation['issues'].append(f'Mesh too large ({max_dimension:.1f}mm max dimension)')
                    validation['recommendations'].append('Scale down the model')

            elif open3d_available and hasattr(mesh, 'compute_vertex_normals'):
                # Basic Open3D validation
                if mesh.is_empty():
                    validation['issues'].append('Mesh is empty')
                    validation['is_valid'] = False

                # Check triangle count
                if len(mesh.triangles) < 4:
                    validation['issues'].append('Mesh has too few triangles')
                    validation['is_valid'] = False

            if validation['issues']:
                validation['is_valid'] = False

        except Exception as e:
            validation['issues'].append(f'Validation failed: {e}')
            validation['is_valid'] = False

        return validation
