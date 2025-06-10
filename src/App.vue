<script setup>
import { ref, onMounted, watch, computed } from 'vue';
import * as echarts from 'echarts';
import OpenSeadragon from 'openseadragon';
import FeatureChart from './components/FeatureChart.vue';
import ImageDisplay from './components/ImageDisplay.vue';
import { loadData } from './utils/featureProcessor.js';

const positiveFeatures = ref([]);
const negativeFeatures = ref([]);
const dziPath = ref(''); // Reactive variable for DZI path
let viewer = null; // To store the OpenSeadragon viewer instance
const representativeFeatures = ref([]);
const selectedImage = ref(null);
const clickedFeatureObject = ref(null); // To store the full clicked feature object
const nearbyImages = ref([]); // Reactive ref for nearby images
const displayMode = ref('all');
const vips = ["201465833_HE_3","201466840_HE_3","201503063_HE_3","201517427_HE_3"]
// Function to find nearby images based on 2D coordinates of the clickedFeature
const findNearbyImages = (clickedFeature, allSourceFeatures, count) => {
  if (!clickedFeature || !clickedFeature.value || !allSourceFeatures || allSourceFeatures.length === 0) {
    return [];
  }

  const targetCoords = clickedFeature.value; // t-SNE coordinates

  const distances = allSourceFeatures
    .filter(feature => feature.id !== clickedFeature.id && feature.value) // Ensure feature has 'value' (t-SNE coords)
    .map((feature) => {
      const dx = feature.value[0] - targetCoords[0];
      const dy = feature.value[1] - targetCoords[1];
      const distSq = dx * dx + dy * dy;
      return { distSq, image: feature.image, id: feature.id }; // Use feature.image
    })
    .filter(item => item !== null);

  distances.sort((a, b) => a.distSq - b.distSq);
  
  const finalImages = distances.slice(0, count).map(item => item.image).filter(img => img);
  console.log('Nearby images calculated in App.vue:', finalImages);
  return finalImages;
};
function getRandomElement(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return undefined;
  const randomIndex = Math.floor(Math.random() * arr.length);
  return arr[randomIndex];
}
const handleFeatureClick = (feature) => {
  console.log('Feature clicked in App.vue:', feature);
  selectedImage.value = feature.image;
  // Placeholder for finding nearby images based on the clicked feature
  // Use representativeFeatures which have the t-SNE coordinates ('value' property)
  nearbyImages.value = findNearbyImages(feature, representativeFeatures.value, 5); // Find 5 nearby images

  // DZI path logic removed as it's not in the new data structure
  // If DZI functionality is needed, it will require a new data field and handling.
  viewer.open(`http://10.86.100.29/vips/zheyi_liver/vips/${getRandomElement(vips)}.dzi`)
};

onMounted(async () => {
  const data = await loadData();
  positiveFeatures.value = data.positiveFeatures;
  negativeFeatures.value = data.negativeFeatures;
  representativeFeatures.value = data.representativeFeatures; // Assign representativeFeatures
  dziPath.value = "http://10.86.100.29/vips/zheyi_liver/vips/201456317_HE_3.dzi"
  // Initialize OpenSeadragon
  viewer = OpenSeadragon({
    id: "openseadragon-viewer",
    prefixUrl: "https://openseadragon.github.io/openseadragon/images/", // Default icons
    tileSources: dziPath.value, // Initial DZI path (can be empty)
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    // You might want to add more OSD options here
  });
      // 创建一个 div 作为矩形
  var rect = document.createElement('div');
  rect.style.border = '2px solid red';
  rect.style.position = 'absolute';
  rect.style.width = '100px';
  rect.style.height = '100px';

  // 添加到 OpenSeadragon 查看器中
  viewer.addOverlay({
    element: rect,
    location: new OpenSeadragon.Rect(0.3, 0.3, 0.2, 0.2) // x, y, width, height (normalized)
  });
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
          :selected-image="selectedImage"
          :nearby-images="nearbyImages"
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
  gap: 1rem;
  overflow: hidden; /* Prevent scrollbars on the main content area */
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
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 300px; /* Minimum width for the right column */
  max-width: 100%; /* Ensure it doesn't overflow its parent */
  overflow-y: auto; /* Allow vertical scrolling if content overflows */
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
