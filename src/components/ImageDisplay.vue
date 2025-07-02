<script setup>
import { computed } from 'vue';

const props = defineProps({
  selectedMeta:Object,
  selectedPoint: String,
  nearbyPoints: Array // Now directly receives the computed nearby images
});
const emit = defineEmits(['chart-click']);
// currentNearbyImages is now directly props.nearbyImages
const currentNearbyPoints = computed(() => props.selectedPoint || []);
const heatmap_url = computed(()=>`http://10.86.100.29/vips/zheyi_liver/heat_and_rank/${props.selectedMeta?.slide_id}/heatmap.png`)
const onClickPoint = (point)=>{
  emit('chart-click', point);
}
</script>

<template>
  <div class="image-display-area">
    <!-- Selected Image Display -->
    <div class="selected-image-container">
      <h3>当前 slide 热力图</h3>
      <div v-if="heatmap_url">
        <img :src="heatmap_url" alt="Selected Feature Image">
      </div>
      <div v-else>
        <p>点击图中的点以显示图像</p>
      </div>
    </div>
    <div class="selected-image-container">
      <h3>选定图像</h3>
      <div @click="onClickPoint(selectedPoint)" style="cursor: pointer;" v-if="selectedPoint">
        {{selectedPoint.slide_id }} {{ selectedPoint.patch_coord }}
        <!-- <img :src="selectedImage" alt="Selected Feature Image"> -->
      </div>
      <div v-else>
        <p>点击图中的点以显示图像</p>
      </div>
    </div>
    <!-- Nearby Images Display -->
    <div class="nearby-images-container">
      <h3>选定点附近图像</h3>
      <div v-if="props.nearbyPoints && props.nearbyPoints.length > 0" class="nearby-images">
        <div @click="onClickPoint(point)" v-for="(point, index) in props.nearbyPoints" :key="`${point}-${index}`" style="cursor: pointer;">
            {{ point.slide_id }} {{ point.patch_coord }}
        </div>
      </div>
      <div v-else>
        <p>点击图中的点以显示附近图像</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-display-area {
  flex: 2; /* Give image area reasonable space (e.g., ~40%) */
  min-width: 350px;
  display: flex;
  flex-direction: column;
  gap: 25px; /* Increased gap */
}

.selected-image-container,
.nearby-images-container {
  border: 1px solid #dee2e6; /* Lighter border */
  padding: 20px; /* Increased padding */
  border-radius: 6px;
  background-color: #ffffff; /* White background */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
}

.selected-image-container {
    min-height: 280px; /* Ensure minimum height */
    justify-content: center; /* Center content vertically */
}

.nearby-images-container {
    flex-grow: 1; /* Allow nearby container to grow */
}

h3 {
  text-align: center;
  margin-top: 0;
  margin-bottom: 20px; /* Increased spacing */
  color: #495057; /* Slightly lighter heading color */
  font-weight: 500;
}

.selected-image-container img {
  display: block;
  max-width: 100%;
  max-height: 200px; /* Adjust max height for selected image */
  margin: 15px auto; /* Increased margin, center the image */
  border: 1px solid #e9ecef; /* Very light border */
  border-radius: 4px;
  object-fit: contain;
}

.selected-image-container p,
.nearby-images-container p {
    text-align: center;
    color: #6c757d; /* Softer text color */
    margin-top: auto; /* Push text to bottom if no image */
    margin-bottom: auto;
    font-style: italic;
}

.nearby-images {

  cursor: pointer;
}

/* Custom scrollbar for nearby images */
.nearby-images::-webkit-scrollbar {
  width: 6px;
}

.nearby-images::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.nearby-images::-webkit-scrollbar-thumb {
  background: #ced4da;
  border-radius: 3px;
}

.nearby-images::-webkit-scrollbar-thumb:hover {
  background: #adb5bd;
}

.nearby-images img {
  width: 100%;
  height: 100px; /* Fixed height for consistency */
  object-fit: cover; /* Crop images nicely */
  border: 1px solid #dee2e6; /* Lighter border */
  border-radius: 4px; /* Slightly more rounded */
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.nearby-images img:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>