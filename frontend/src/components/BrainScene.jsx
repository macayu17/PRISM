import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useLoader, extend } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { TextureLoader } from 'three/src/loaders/TextureLoader';
import { shaderMaterial, Stars, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// --- Custom X-Ray Shader Material ---
const XRayMaterial = shaderMaterial(
    {
        c: 1.0,
        p: 1.4,
        glowColor: new THREE.Color(0x84ccff),
        viewVector: new THREE.Vector3(0, 0, 0),
        lightningTexture: null,
        offsetY: 0.0,
        uTime: 0.0,
        // Heatmap Uniforms: x, y, z, intensity
        uHeatmap1: new THREE.Vector4(0, 0, 0, 0), // Tremor
        uHeatmap2: new THREE.Vector4(0, 0, 0, 0), // Rigidity
        uHeatmap3: new THREE.Vector4(0, 0, 0, 0), // Bradykinesia
        uHeatmap4: new THREE.Vector4(0, 0, 0, 0), // Postural Instability
    },
    // Vertex Shader
    `
    uniform vec3 viewVector;
    uniform float c;
    uniform float p;
    varying float intensity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      vec3 vNormal = normalize( normalMatrix * normal );
      vec3 vView = normalize( viewVector - position );
      intensity = pow(c - abs(dot(vNormal, vView)), p);
      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
    }
  `,
    // Fragment Shader
    `
    uniform vec3 glowColor;
    uniform sampler2D lightningTexture;
    varying float intensity;
    varying vec2 vUv;
    varying vec3 vPosition;
    uniform float offsetY;
    uniform float uTime;

    uniform vec4 uHeatmap1; 
    uniform vec4 uHeatmap2;
    uniform vec4 uHeatmap3;
    uniform vec4 uHeatmap4;
    
    void main() {
      vec2 uv = vUv;
      uv.y += offsetY;
      
      vec4 tex = texture2D(lightningTexture, vec2(uv.x, uv.y + uTime * 0.2));
      vec3 electric = tex.rgb;
      
      // --- Heatmap Logic ---
      // Centers (approximate model coordinates)
      // Radius = 15.0 - 25.0
      
      float d1 = distance(vPosition, uHeatmap1.xyz);
      float h1 = uHeatmap1.w * smoothstep(25.0, 5.0, d1); // Tremor (Center)

      float d2 = distance(vPosition, uHeatmap2.xyz);
      float h2 = uHeatmap2.w * smoothstep(25.0, 5.0, d2); // Rigidity (Top/Motor)

      float d3 = distance(vPosition, uHeatmap3.xyz);
      float h3 = uHeatmap3.w * smoothstep(30.0, 5.0, d3); // Brady (Back/Cerebellum)
      
      float d4 = distance(vPosition, uHeatmap4.xyz);
      float h4 = uHeatmap4.w * smoothstep(30.0, 5.0, d4); // Postural (Brainstem/Lower)

      float totalHeat = clamp(h1 + h2 + h3 + h4, 0.0, 1.2);

      // Heat Color: Red/Orange
      vec3 heatColor = vec3(1.0, 0.3, 0.1);

      // Force some visibility even if texture is dark
      float alpha = clamp(intensity, 0.2, 1.0); 
      
      vec3 baseColor = glowColor * intensity + electric * intensity * 2.0;
      
      // Mix base with heat
      // If totalHeat > 0, we blend towards heatColor * intensity
      vec3 finalColor = mix(baseColor, heatColor * (intensity + 0.5), totalHeat);

      gl_FragColor = vec4( finalColor, alpha ); 
    }
  `
);

extend({ XRayMaterial });

function BrainModel({ symptomData }) {
    const obj = useLoader(OBJLoader, '/models/BrainUVs.obj');
    const lightningMap = useLoader(TextureLoader, '/textures/brainXRayLight.png');
    const brainGroup = useRef();
    const xRayRef = useRef();

    // Extract geometry for Points
    const pointsGeometry = useMemo(() => {
        let combinedGeometry = new THREE.BufferGeometry();
        let vertices = [];

        obj.traverse((child) => {
            if (child.isMesh) {
                const pos = child.geometry.attributes.position;
                for (let i = 0; i < pos.count; i++) {
                    vertices.push(pos.getX(i), pos.getY(i), pos.getZ(i));
                }
            }
        });

        combinedGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        return combinedGeometry;
    }, [obj]);

    // Extract geometry for Mesh (X-Ray shell)
    const meshGeometry = useMemo(() => {
        let combinedGeometry = new THREE.BufferGeometry();
        let vertices = [];
        let normals = [];
        let uvs = [];

        obj.traverse((child) => {
            if (child.isMesh) {
                const pos = child.geometry.attributes.position;
                const norm = child.geometry.attributes.normal;
                const uv = child.geometry.attributes.uv;

                if (pos) {
                    for (let i = 0; i < pos.count; i++) {
                        vertices.push(pos.getX(i), pos.getY(i), pos.getZ(i));
                        if (norm) normals.push(norm.getX(i), norm.getY(i), norm.getZ(i));
                        else normals.push(0, 1, 0);

                        if (uv) uvs.push(uv.getX(i), uv.getY(i));
                        else uvs.push(0, 0);
                    }
                }
            }
        });

        combinedGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        combinedGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        combinedGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

        return combinedGeometry;
    }, [obj]);


    useFrame((state) => {
        const time = state.clock.getElapsedTime();
        if (xRayRef.current) {
            xRayRef.current.material.uniforms.uTime.value = time;
            xRayRef.current.material.uniforms.viewVector.value = state.camera.position;
            xRayRef.current.material.uniforms.offsetY.value = Math.sin(time * 0.5) * 0.1;

            // --- Update Heatmap Uniforms ---
            if (symptomData) {
                // Helper: Normalize 0-4 to 0.0-1.0
                const getInt = (val) => (val ? parseFloat(val) / 4.0 : 0.0);

                // 1. Tremor -> Basal Ganglia (Deep Center): (0, 0, 0)
                xRayRef.current.material.uniforms.uHeatmap1.value.set(0, 0, 0, getInt(symptomData.sym_tremor));

                // 2. Rigidity -> Motor Cortex (Top): (0, 25, 0)
                xRayRef.current.material.uniforms.uHeatmap2.value.set(0, 25, 0, getInt(symptomData.sym_rigid));

                // 3. Bradykinesia -> Cerebellum (Back/Bottom): (0, -10, -20)
                xRayRef.current.material.uniforms.uHeatmap3.value.set(0, -10, -20, getInt(symptomData.sym_brady));

                // 4. Postural Instability -> Brainstem (Bottom Center): (0, -20, 5)
                xRayRef.current.material.uniforms.uHeatmap4.value.set(0, -20, 5, getInt(symptomData.sym_posins));
            } else {
                // Reset if no data
                xRayRef.current.material.uniforms.uHeatmap1.value.w = 0;
                xRayRef.current.material.uniforms.uHeatmap2.value.w = 0;
                xRayRef.current.material.uniforms.uHeatmap3.value.w = 0;
                xRayRef.current.material.uniforms.uHeatmap4.value.w = 0;
            }
        }
    });

    return (
        <group ref={brainGroup} scale={[0.8, 0.8, 0.8]}>

            {/* 1. X-Ray Shell */}
            <mesh ref={xRayRef} geometry={meshGeometry}>
                {/* @ts-ignore */}
                <xRayMaterial
                    attach="material"
                    transparent
                    depthWrite={false}
                    side={THREE.DoubleSide}
                    blending={THREE.AdditiveBlending}
                    lightningTexture={lightningMap}
                    glowColor={new THREE.Color('#38bdf8')}
                    c={0.3}
                    p={2.0}
                />
            </mesh>

            {/* 2. Particles */}
            <points>
                <bufferGeometry attach="geometry" {...pointsGeometry} />
                <pointsMaterial
                    size={1.5}
                    color="#bae6fd"
                    transparent
                    opacity={0.8}
                    sizeAttenuation
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </points>

        </group>
    );
}

export default function BrainScene({ symptomData }) {
    return (
        <div className="h-full w-full">
            <Canvas
                camera={{ position: [0, 0, 140], fov: 45, near: 0.1, far: 2000 }}
                gl={{ alpha: true, antialias: true, preserveDrawingBuffer: true }}
            >
                <ambientLight intensity={1.5} />
                <pointLight position={[50, 50, 50]} intensity={2} color="#ffffff" />
                <pointLight position={[-50, -50, 50]} intensity={2} color="#00ffff" />

                <React.Suspense fallback={null}>
                    <BrainModel symptomData={symptomData} />
                </React.Suspense>

                {/* Interaction controls */}
                <OrbitControls
                    enableZoom={true}
                    enablePan={false}
                    autoRotate={true}
                    autoRotateSpeed={0.5}
                    minDistance={50}
                    maxDistance={500}
                />

                <Stars radius={300} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
            </Canvas>
        </div>
    );
}
