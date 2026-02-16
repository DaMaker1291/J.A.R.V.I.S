import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface SpatialAnchor {
  id: string;
  position: [number, number, number];
  color: string;
}

const SpatialAnchor: React.FC<{ anchor: SpatialAnchor }> = ({ anchor }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime) * 0.1;
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <Sphere ref={meshRef} args={[0.1]} position={anchor.position}>
      <meshStandardMaterial color={anchor.color} emissive={anchor.color} emissiveIntensity={0.2} />
    </Sphere>
  );
};

const SpatialMapping: React.FC = () => {
  // Mock spatial anchors - in real implementation, fetch from YOLOv8 API
  const anchors: SpatialAnchor[] = [
    { id: 'phone', position: [0, 0, 0], color: '#00E5FF' },
    { id: 'object1', position: [2, 1, -1], color: '#FF3D00' },
    { id: 'object2', position: [-1.5, 0.5, 1], color: '#FF8C00' },
  ];

  return (
    <div className="h-full w-full">
      <Canvas camera={{ position: [5, 5, 5], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        {anchors.map((anchor) => (
          <SpatialAnchor key={anchor.id} anchor={anchor} />
        ))}
        {/* Room boundaries */}
        <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[10, 10]} />
          <meshStandardMaterial color="#1a1a1a" transparent opacity={0.3} />
        </mesh>
      </Canvas>
    </div>
  );
};

export default SpatialMapping;
