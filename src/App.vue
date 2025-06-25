<script setup>
import { ref, onMounted, watch, computed } from 'vue';
import * as echarts from 'echarts';
import OpenSeadragon from 'openseadragon';
import FeatureChart from './components/FeatureChart.vue';
import ImageDisplay from './components/ImageDisplay.vue';
import { loadData } from './utils/featureProcessor.js';

const positiveFeatures = ref([]);
const negativeFeatures = ref([]);
const slice_meta = ref([])
const dziPath = ref(''); // Reactive variable for DZI path
let viewer = null; // To store the OpenSeadragon viewer instance
const representativeFeatures = ref([]);
const selectedPoint = ref(null);
const selectedMeta = ref(null)
const clickedFeatureObject = ref(null); // To store the full clicked feature object
const nearbyPoints = ref([]); // Reactive ref for nearby images
const displayMode = ref('all');
// Function to find nearby images based on 2D coordinates of the clickedFeature
const findNearbyPoints = (clickedFeature, allSourceFeatures, count) => {
  if (!clickedFeature || !clickedFeature.UMAP_3d || !allSourceFeatures || allSourceFeatures.length === 0) {
    return [];
  }

  const targetCoords = clickedFeature.UMAP_3d; // t-SNE coordinates

  const distances = allSourceFeatures
    .filter(feature => feature.slide_id !== clickedFeature.slide_id && feature.UMAP_3d) // Ensure feature has 'value' (t-SNE coords)
    .map((feature) => {
      const dx = feature.UMAP_3d[0] - targetCoords[0];
      const dy = feature.UMAP_3d[1] - targetCoords[1];
      const dz = feature.UMAP_3d[2] - targetCoords[2];
      const distSq = dx * dx + dy * dy + dz * dz;
      return { distSq, coord: feature.UMAP_3d, slide_id: feature.slide_id ,patch_coord:feature.patch_coord}; // Use feature.image
    })
    .filter(item => item !== null);

  distances.sort((a, b) => a.distSq - b.distSq);
  
  const finalPoints = distances.slice(0, count)
  console.log('Nearby images calculated in App.vue:', finalPoints);
  console.log(finalPoints)
  return finalPoints;
};
function getRandomElement(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return undefined;
  const randomIndex = Math.floor(Math.random() * arr.length);
  return arr[randomIndex];
}
const handleFeatureClick = (feature) => {
  console.log('Feature clicked in App.vue:', feature);
  selectedPoint.value = feature;

  // Placeholder for finding nearby images based on the clicked feature
  // Use representativeFeatures which have the t-SNE coordinates ('value' property)
  nearbyPoints.value = findNearbyPoints(feature, representativeFeatures.value, 5); // Find 5 nearby images

  // DZI path logic removed as it's not in the new data structure
  // If DZI functionality is needed, it will require a new data field and handling.
  viewer.open(`http://10.86.100.29/vips/zheyi_liver/vips/${feature.slide_id}.dzi`)
  // console.log("lhm=====",feature)
  selectedMeta.value= Object.assign(slice_meta.value[feature.slide_id],{"slide_id":feature.slide_id})
  const coord = feature.patch_coord
  const size = selectedMeta.value.size
  const viewportPoint = viewer.viewport.imageToViewportCoordinates(Math.random(),Math.random())
  viewer.addHandler('open', () => { 
    viewer.viewport.zoomTo(20)
    viewer.viewport.panTo(viewportPoint)
  })


};

onMounted(async () => {
  const data = await loadData();
  positiveFeatures.value = data.positiveFeatures;
  negativeFeatures.value = data.negativeFeatures;
  representativeFeatures.value = data.representativeFeatures; // Assign representativeFeatures
  slice_meta.value = data.slide_meta;
  const random_point = getRandomElement(positiveFeatures.value)
  selectedMeta.value= Object.assign(slice_meta.value[random_point.slide_id],{"slide_id":random_point.slide_id})
  console.log("=======",random_point,selectedMeta.value)

  dziPath.value = `http://10.86.100.29/vips/zheyi_liver/vips/${random_point.slide_id}.dzi`
  // Initialize OpenSeadragon
  viewer = OpenSeadragon({
    id: "openseadragon-viewer",
    prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/", // Default icons
    tileSources: dziPath.value, // Initial DZI path (can be empty)
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    // You might want to add more OSD options here
  });
  const coord = random_point.patch_coord
  const size = selectedMeta.value.size

  const viewportPoint = viewer.viewport.imageToViewportCoordinates(Math.random(),Math.random())
  viewer.addHandler('open', () => { 
    viewer.viewport.zoomTo(20)
    console.log("lhm===",viewportPoint)
    viewer.viewport.panTo(viewportPoint)
    console.log(coord,size)
  })
});

// Watch for changes in displayMode to pass to FeatureChart
// FeatureChart itself will handle its internal update logic based on this prop

</script>

<template>
  <div id="app-container">
    <header>
      <h1>PathMeans: Interactive Visualization</h1>
      <div class="controls">
        <span>Display Mode:</span>
        <label>
          <input type="radio" v-model="displayMode" value="all"> All
        </label>
        <label>
          <input type="radio" v-model="displayMode" value="positive"> Positive Only
        </label>
        <label>
          <input type="radio" v-model="displayMode" value="negative"> Negative Only
        </label>
      </div>
    </header>
    <main class="main-content-grid">
      <div id="openseadragon-viewer" class="osd-container"></div>
      <div class="chart-and-image-container">
        <FeatureChart
          :representativeFeatures="representativeFeatures"
          :display-mode="displayMode"
          @chart-click="handleFeatureClick"
        />
        <ImageDisplay
          :selected-point="selectedPoint"
          :nearby-points="nearbyPoints"
          :selected-meta="selectedMeta"
        />
      </div>
    </main>
  </div>
</template>

<style scoped>
#app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  font-family: 'Arial', sans-serif;
  background-color: #f4f4f9;
  color: #333;
}

header {
  background-color: #607D8B; /* Material Blue Grey */
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

header h1 {
  margin: 0;
  font-size: 1.8rem;
}

.controls span {
  margin-right: 0.5rem;
  font-size: 1rem;
}

.controls label {
  margin-left: 1rem;
  cursor: pointer;
}

.controls input[type="radio"] {
  margin-right: 0.3rem;
}

.main-content-grid {
  display: flex;
  flex-grow: 1;
  padding: 1rem;
  /* gap: 1rem; */
  overflow: scroll; /* Prevent scrollbars on the main content area */
  max-width: 100vw; /* Ensure it doesn't exceed viewport width */
  box-sizing: border-box; /* Include padding and border in the element's total width and height */
}

.osd-container {
  flex: 2; /* Takes 2/3 of the space */
  background-color: #e0e0e0; /* Placeholder background */
  border: 1px solid #ccc;
  min-height: 300px; /* Minimum height */
}

.chart-and-image-container {
  flex: 1; /* Takes 1/3 of the space */
  /* display: flex; */
  /* flex-direction: column; */
  gap: 1rem;
  min-width: 300px; /* Minimum width for the right column */
  max-width: 100%; /* Ensure it doesn't overflow its parent */
  overflow-y: scroll;
  /* overflow-y: auto; Allow vertical scrolling if content overflows */
}

/* Responsive adjustments */
@media (max-width: 1024px) { /* Adjusted breakpoint for better layout on medium screens */
  .main-content-grid {
    flex-direction: column;
  }
  .osd-container {
    flex: 1; /* Full width on smaller screens */
    margin-bottom: 1rem; /* Add some space when stacked */
  }
  .chart-and-image-container {
    flex: 1; /* Full width on smaller screens */
  }
}

@media (max-width: 768px) {
  header {
    flex-direction: column;
    padding: 1rem;
  }
  header h1 {
    margin-bottom: 0.5rem;
  }
}
</style>
