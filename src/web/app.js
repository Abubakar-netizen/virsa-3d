// Virsa3D - Interactive 3D Heritage Visualizer Engine
let scene, camera, renderer, controls;
let currentPointsMesh = null;
let masterPositions = null;
let masterColors = null;
let masterLabels = null;
let currentPointSize = 0.18;
let autoRotate = false;
let currentModelType = "proposed";

// Initialize Viewport on Load
window.addEventListener("DOMContentLoaded", () => {
  initThree();
  loadSiteInfo();
  loadMetricsComparison();
  loadModel("proposed");
  window.addEventListener("resize", onWindowResize);
});

function initThree() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b14);
  scene.fog = new THREE.FogExp2(0x070b14, 0.015);

  // Camera
  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.set(0, -28, 16);
  camera.up.set(0, 0, 1); // Z is up

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;
  controls.target.set(0, 0, 2.5);

  // Lighting & Grid Helper
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xfffaed, 0.8);
  dirLight.position.set(20, 20, 40);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(40, 40, 0x06b6d4, 0x1e293b);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // Render Loop
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  if (autoRotate && controls) {
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.5;
  } else if (controls) {
    controls.autoRotate = false;
  }
  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// Load 3D Gaussian / PLY Model
function loadModel(modelType) {
  currentModelType = modelType;
  showLoading(true, `Streaming ${modelType.toUpperCase()} 3D Scene...`);

  const loader = new THREE.PLYLoader();
  const url = `/api/models/${modelType}`;

  loader.load(
    url,
    function (geometry) {
      if (currentPointsMesh) {
        scene.remove(currentPointsMesh);
        currentPointsMesh.geometry.dispose();
        currentPointsMesh.material.dispose();
      }

      geometry.computeVertexNormals();

      // Extract attributes
      const posAttr = geometry.getAttribute("position");
      const colAttr = geometry.getAttribute("color");
      const totalPoints = posAttr.count;

      masterPositions = new Float32Array(posAttr.array);
      if (colAttr) {
        masterColors = new Float32Array(colAttr.array);
      } else {
        masterColors = new Float32Array(totalPoints * 3);
        for (let i = 0; i < totalPoints * 3; i++) masterColors[i] = 0.8;
      }

      // Infer / parse semantic labels based on color/height if label attribute is absent
      masterLabels = new Int32Array(totalPoints);
      for (let i = 0; i < totalPoints; i++) {
        const z = masterPositions[i * 3 + 2];
        const r = masterColors[i * 3];
        const g = masterColors[i * 3 + 1];
        const b = masterColors[i * 3 + 2];

        // Classification matching:
        if (g > r && g > b && z > 0.3) {
          masterLabels[i] = 3; // Vegetation
        } else if (z > 0.4 || (Math.abs(masterPositions[i * 3]) < 5.5 && Math.abs(masterPositions[i * 3 + 1]) < 5.5)) {
          masterLabels[i] = 2; // Structure / Temple
        } else {
          masterLabels[i] = 1; // Ground
        }
      }

      // Create initial material & points
      const material = new THREE.PointsMaterial({
        size: currentPointSize,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.95
      });

      currentPointsMesh = new THREE.Points(geometry, material);
      scene.add(currentPointsMesh);

      // Update UI Telemetry
      document.getElementById("tele-points").innerText = totalPoints.toLocaleString();
      document.getElementById("tele-state").innerText = modelType.toUpperCase();

      updateSemanticFilters();
      showLoading(false);
    },
    function (xhr) {
      if (xhr.lengthComputable) {
        const percent = Math.round((xhr.loaded / xhr.total) * 100);
        document.getElementById("loading-text").innerText = `Downloading 3D Model: ${percent}%`;
      }
    },
    function (error) {
      console.error("Error loading PLY model:", error);
      showLoading(true, "Error loading 3D model. Check backend server.");
    }
  );
}

// Semantic Filter Toggles
function updateSemanticFilters() {
  if (!currentPointsMesh || !masterPositions) return;

  const showStructure = document.getElementById("layer-structure").checked;
  const showGround = document.getElementById("layer-ground").checked;
  const showVegetation = document.getElementById("layer-vegetation").checked;

  const validIndices = [];
  const total = masterLabels.length;

  for (let i = 0; i < total; i++) {
    const lbl = masterLabels[i];
    if (lbl === 2 && showStructure) validIndices.push(i);
    else if (lbl === 1 && showGround) validIndices.push(i);
    else if (lbl === 3 && showVegetation) validIndices.push(i);
  }

  const filteredPositions = new Float32Array(validIndices.length * 3);
  const filteredColors = new Float32Array(validIndices.length * 3);

  for (let idx = 0; idx < validIndices.length; idx++) {
    const src = validIndices[idx];
    filteredPositions[idx * 3] = masterPositions[src * 3];
    filteredPositions[idx * 3 + 1] = masterPositions[src * 3 + 1];
    filteredPositions[idx * 3 + 2] = masterPositions[src * 3 + 2];

    filteredColors[idx * 3] = masterColors[src * 3];
    filteredColors[idx * 3 + 1] = masterColors[src * 3 + 1];
    filteredColors[idx * 3 + 2] = masterColors[src * 3 + 2];
  }

  const geom = currentPointsMesh.geometry;
  geom.setAttribute("position", new THREE.BufferAttribute(filteredPositions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(filteredColors, 3));
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
  geom.setDrawRange(0, validIndices.length);

  document.getElementById("tele-points").innerText = validIndices.length.toLocaleString();
}

// Model Switch Handler
function switchModel(type) {
  document.getElementById("btn-baseline").classList.toggle("active", type === "baseline");
  document.getElementById("btn-proposed").classList.toggle("active", type === "proposed");
  loadModel(type);
}

// Display Controls
function changePointSize(val) {
  currentPointSize = parseFloat(val);
  document.getElementById("val-point-size").innerText = val;
  if (currentPointsMesh && currentPointsMesh.material) {
    currentPointsMesh.material.size = currentPointSize;
  }
}

function toggleAutoRotate() {
  autoRotate = !autoRotate;
  const btn = document.getElementById("btn-autorotate");
  btn.innerText = `Auto-Rotate: ${autoRotate ? "ON" : "OFF"}`;
  btn.classList.toggle("active", autoRotate);
}

function resetCamera() {
  if (camera && controls) {
    camera.position.set(0, -28, 16);
    camera.up.set(0, 0, 1);
    controls.target.set(0, 0, 2.5);
    controls.update();
  }
}

function togglePanel(id) {
  const panel = document.getElementById(id);
  panel.classList.toggle("active");
}

function showLoading(show, text = "") {
  const overlay = document.getElementById("loading-overlay");
  if (show) {
    overlay.classList.remove("hidden");
    if (text) document.getElementById("loading-text").innerText = text;
  } else {
    overlay.classList.add("hidden");
  }
}

// Fetch Backend Site Info
async function loadSiteInfo() {
  try {
    const res = await fetch("/api/site-info");
    const data = await res.json();
    document.getElementById("meta-site-name").innerText = data.site_name;
    document.getElementById("meta-period").innerText = data.period;
    document.getElementById("meta-region").innerText = `📍 ${data.region}`;
    document.getElementById("meta-lat").innerText = `${data.gps.latitude}° N`;
    document.getElementById("meta-lon").innerText = `${data.gps.longitude}° E`;
    document.getElementById("meta-alt").innerText = `${data.gps.altitude_min_m}m - ${data.gps.altitude_max_m}m`;
    document.getElementById("meta-images").innerText = `${data.drone_survey.total_images} UAV images`;
    document.getElementById("meta-drone").innerText = data.drone_survey.platform;

    const featList = document.getElementById("meta-features");
    if (featList && data.features) {
      featList.innerHTML = data.features.map(f => `<li>${f}</li>`).join("");
    }
  } catch (e) {
    console.warn("Could not fetch site info:", e);
  }
}

// Fetch Backend Metrics Comparison
async function loadMetricsComparison() {
  try {
    const res = await fetch("/api/metrics/comparison");
    const data = await res.json();
    if (data.baseline && data.proposed) {
      const b = data.baseline.macro_average || {};
      const p = data.proposed.macro_average || {};
      const tbody = document.getElementById("metrics-table-body");
      if (tbody) {
        tbody.innerHTML = `
          <tr>
            <td><strong>Ground</strong></td>
            <td>${(data.baseline.per_class.ground.iou * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.ground.iou * 100).toFixed(1)}%</td>
            <td>${(data.baseline.per_class.ground.f1 * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.ground.f1 * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <td><strong>Structure</strong></td>
            <td>${(data.baseline.per_class.structure.iou * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.structure.iou * 100).toFixed(1)}%</td>
            <td>${(data.baseline.per_class.structure.f1 * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.structure.f1 * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <td><strong>Vegetation</strong></td>
            <td>${(data.baseline.per_class.vegetation.iou * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.vegetation.iou * 100).toFixed(1)}%</td>
            <td>${(data.baseline.per_class.vegetation.f1 * 100).toFixed(1)}%</td>
            <td class="prop">${(data.proposed.per_class.vegetation.f1 * 100).toFixed(1)}%</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>Mean (Macro)</strong></td>
            <td>${(b.mean_iou * 100).toFixed(1)}%</td>
            <td class="prop">${(p.mean_iou * 100).toFixed(1)}%</td>
            <td>${(b.mean_f1 * 100).toFixed(1)}%</td>
            <td class="prop">${(p.mean_f1 * 100).toFixed(1)}%</td>
          </tr>
        `;
      }
    }
  } catch (e) {
    console.warn("Could not fetch metrics comparison:", e);
  }
}
