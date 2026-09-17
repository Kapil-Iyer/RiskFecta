import * as THREE from "three";

export interface UniverseSceneNode {
  ticker: string;
  sector: "Information Technology" | "Financials";
}

export interface HoverInfo {
  ticker: string;
  x: number;
  y: number;
}

interface SceneOptions {
  reducedMotion: boolean;
  onHover: (info: HoverInfo | null) => void;
}

export interface SceneHandle {
  dispose: () => void;
}

const IT_COLOR = new THREE.Color("#38bdf8"); // matches --sector-it (theme.ts)
const FIN_COLOR = new THREE.Color("#d9a441"); // matches --sector-fin (theme.ts)

/** Evenly distributes `count` points on a sphere surface (Fibonacci sphere)
 * — a deterministic layout for legibility, not a physics/force simulation
 * and not an encoding of any real relationship between tickers. */
function fibonacciSpherePoints(count: number, radius: number): THREE.Vector3[] {
  const points: THREE.Vector3[] = [];
  if (count === 0) return points;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = count === 1 ? 0 : 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;
    points.push(new THREE.Vector3(Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius));
  }
  return points;
}

/**
 * A restrained 3D visualization of the real RiskFecta universe: two loose
 * clusters (Information Technology / Financials), one node per real
 * ticker. This deliberately draws NO connecting lines or edges between
 * nodes — there is no computed correlation/covariance/relationship this
 * visualization is authorized to represent, and adding lines would imply
 * one that was never computed (see PRD.md UX principles).
 */
export function createUniverseScene(
  container: HTMLElement,
  nodes: UniverseSceneNode[],
  { reducedMotion, onHover }: SceneOptions,
): SceneHandle {
  const width = container.clientWidth || 1;
  const height = container.clientHeight || 1;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0, 13);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  const group = new THREE.Group();
  scene.add(group);
  scene.add(new THREE.AmbientLight(0xffffff, 1));

  const itNodes = nodes.filter((n) => n.sector === "Information Technology");
  const finNodes = nodes.filter((n) => n.sector === "Financials");
  const itPositions = fibonacciSpherePoints(itNodes.length, 3.2).map((p) => p.add(new THREE.Vector3(-4, 0, 0)));
  const finPositions = fibonacciSpherePoints(finNodes.length, 3.2).map((p) => p.add(new THREE.Vector3(4, 0, 0)));

  const geometry = new THREE.SphereGeometry(0.16, 12, 12);
  const meshes: THREE.Mesh[] = [];

  function addNode(ticker: string, color: THREE.Color, position: THREE.Vector3) {
    const material = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData.ticker = ticker;
    group.add(mesh);
    meshes.push(mesh);
  }

  itNodes.forEach((n, i) => addNode(n.ticker, IT_COLOR, itPositions[i]));
  finNodes.forEach((n, i) => addNode(n.ticker, FIN_COLOR, finPositions[i]));

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hoveredMesh: THREE.Mesh | null = null;

  function setHover(mesh: THREE.Mesh | null, x: number, y: number) {
    if (hoveredMesh === mesh) return;
    if (hoveredMesh) hoveredMesh.scale.setScalar(1);
    hoveredMesh = mesh;
    if (mesh) {
      mesh.scale.setScalar(1.9);
      onHover({ ticker: mesh.userData.ticker as string, x, y });
    } else {
      onHover(null);
    }
  }

  function handlePointerMove(event: PointerEvent) {
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    pointer.x = (x / rect.width) * 2 - 1;
    pointer.y = -(y / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(meshes);
    setHover(hits.length > 0 ? (hits[0].object as THREE.Mesh) : null, x, y);
  }

  function handlePointerLeave() {
    setHover(null, 0, 0);
  }

  container.addEventListener("pointermove", handlePointerMove);
  container.addEventListener("pointerleave", handlePointerLeave);

  const resizeObserver = new ResizeObserver(() => {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resizeObserver.observe(container);

  let frameId: number | null = null;
  const clock = new THREE.Clock();
  let disposed = false;

  function renderFrame() {
    if (disposed) return;
    const delta = clock.getDelta();
    if (!reducedMotion) {
      group.rotation.y += delta * 0.12;
    }
    renderer.render(scene, camera);
    if (!reducedMotion) {
      frameId = requestAnimationFrame(renderFrame);
    }
  }

  renderFrame(); // always render at least one frame; looped only when motion is allowed

  return {
    dispose() {
      disposed = true;
      if (frameId !== null) cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      container.removeEventListener("pointermove", handlePointerMove);
      container.removeEventListener("pointerleave", handlePointerLeave);
      meshes.forEach((m) => (m.material as THREE.Material).dispose());
      geometry.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    },
  };
}
