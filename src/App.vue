<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'; // Consolidate imports, add watch
import * as echarts from 'echarts';

// Placeholder for data
const positiveFeatures = ref([]);
const negativeFeatures = ref([]);
const representativeFeatures = ref([]); // Stores features for charting
const selectedImage = ref(null);
const chartInstance = ref(null);
const nearbyImages = ref([]);
const displayMode = ref('all'); // 'all', 'positive', 'negative'

// Random data generator with clustered distribution, increased centers and overlap
const generateRandomFeatures = (count, type) => {
  // Define cluster centers closer together to encourage overlap
  const centers = type === 'positive' ?
    [
      [0.3, 0.3, 0.1], // Closer to origin and each other
      [0.5, 0.1, 0.3],
      [0.1, 0.5, 0.2],
      [0.4, -0.2, 0.2], // Increased overlap potential
      [-0.2, 0.4, 0.0]
    ] :
    [
      [-0.3, -0.3, -0.1], // Closer to origin and each other
      [-0.5, -0.1, -0.3],
      [-0.1, -0.5, -0.2],
      [-0.4, 0.2, -0.2], // Increased overlap potential
      [0.2, -0.4, 0.0]
    ];

  // Generate random points around centers
  const features = [];
  let images = ['/mock_images/pos_1.png', '/mock_images/pos_2.png', '/mock_images/pos_3.png', '/mock_images/pos_4.png', '/mock_images/pos_5.png'];
  
  if(type === 'negative') {
    images = ['/mock_images/neg_1.png', '/mock_images/neg_2.png', '/mock_images/neg_3.png', '/mock_images/neg_4.png', '/mock_images/neg_5.png'];
  }

  for (let i = 0; i < count; i++) {
    const center = centers[Math.floor(Math.random() * centers.length)];
    const feature = {
      id: `${type}_${i+1}`,
      image: images[i % images.length],
      filename: `original_${type}_${i+1}.dat`,
      type: type,
      feature_vector: [
        center[0] + (Math.random() * 1.0 - 0.5), // Slightly increased spread range [-0.5, 0.5]
        center[1] + (Math.random() * 1.0 - 0.5),
        center[2] + (Math.random() * 1.0 - 0.5)
      ]
    };
    features.push(feature);
  }
  
  return features;
};

// Helper function for Euclidean distance squared
const euclideanDistSq = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return sum;
};

// Custom t-SNE implementation
const basicTSNE = (data, { perplexity = 30, learningRate = 100, iterations = 500, dims = 2 }) => {
  console.log('Running custom t-SNE implementation...');
  if (!data || data.length === 0) return [];

  const n = data.length;
  if (n <= perplexity) {
    console.warn(`Perplexity ${perplexity} is too high for ${n} data points. Reducing perplexity to ${Math.max(1, Math.floor(n / 3))}.`);
    perplexity = Math.max(1, Math.floor(n / 3));
  }

  // --- High-Dimensional Similarities (P_ij) --- 
  const P = Array(n).fill(0).map(() => Array(n).fill(0));
  const highDimDistances = Array(n).fill(0).map(() => Array(n).fill(0));

  // Precompute high-dimensional distances
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const distSq = euclideanDistSq(data[i].originalValue, data[j].originalValue);
      highDimDistances[i][j] = highDimDistances[j][i] = Math.sqrt(distSq);
    }
  }

  // Calculate conditional probabilities P_j|i with binary search for sigma
  for (let i = 0; i < n; i++) {
    let beta = 1.0;
    let min_beta = -Infinity, max_beta = Infinity;
    let tol = 1e-5;
    let max_tries = 50;
    let H_target = Math.log2(perplexity);
    let P_i = Array(n).fill(0);

    for (let tries = 0; tries < max_tries; tries++) {
      let H_current = 0;
      let sum_Pi = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        P_i[j] = Math.exp(-beta * highDimDistances[i][j] * highDimDistances[i][j]);
        sum_Pi += P_i[j];
      }

      if (sum_Pi === 0) sum_Pi = 1e-12; // Avoid division by zero

      let sum_P_logP = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        let p = P_i[j] / sum_Pi;
        if (p > 1e-7) sum_P_logP += p * Math.log2(p);
      }
      H_current = -sum_P_logP;

      let H_diff = H_current - H_target;
      if (Math.abs(H_diff) < tol) {
        break; // Found suitable beta
      }

      // Adjust beta using binary search
      if (H_diff > 0) {
        min_beta = beta;
        if (max_beta === Infinity) beta *= 2;
        else beta = (beta + max_beta) / 2;
      } else {
        max_beta = beta;
        if (min_beta === -Infinity) beta /= 2;
        else beta = (beta + min_beta) / 2;
      }
    }
    
    // Normalize P_j|i
    let sum_Pi_final = 0;
    for(let j=0; j<n; ++j) sum_Pi_final += P_i[j];
    if (sum_Pi_final === 0) sum_Pi_final = 1e-12;
    for(let j=0; j<n; ++j) P[i][j] = P_i[j] / sum_Pi_final;
  }

  // Symmetrize probabilities P_ij = (P_j|i + P_i|j) / 2n
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let pij = (P[i][j] + P[j][i]) / (2 * n);
      P[i][j] = P[j][i] = Math.max(pij, 1e-12); // Ensure non-zero
    }
  }

  // --- Gradient Descent --- 
  let Y = Array(n).fill(0).map(() => Array(dims).fill(0).map(() => (Math.random() - 0.5) * 0.0001)); // Initialize low-dim points
  let gains = Array(n).fill(0).map(() => Array(dims).fill(1.0));
  let Y_step = Array(n).fill(0).map(() => Array(dims).fill(0.0));
  const min_grad_norm = 1e-7;

  for (let iter = 0; iter < iterations; iter++) {
    // Calculate low-dimensional similarities (Q_ij)
    const lowDimDistSq = Array(n).fill(0).map(() => Array(n).fill(0));
    let sum_q_denom = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const distSq = euclideanDistSq(Y[i], Y[j]);
        const q_val = 1.0 / (1.0 + distSq);
        lowDimDistSq[i][j] = lowDimDistSq[j][i] = q_val;
        sum_q_denom += 2 * q_val;
      }
    }
    if (sum_q_denom === 0) sum_q_denom = 1e-12;

    const Q = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        Q[i][j] = Q[j][i] = Math.max(lowDimDistSq[i][j] / sum_q_denom, 1e-12);
      }
    }

    // Calculate gradient dC/dY
    const dY = Array(n).fill(0).map(() => Array(dims).fill(0.0));
    const momentum = iter < 250 ? 0.5 : 0.8;
    const earlyExaggeration = iter < 100 ? 4.0 : 1.0;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const grad_coeff = earlyExaggeration * (P[i][j] - Q[i][j]) * lowDimDistSq[i][j];
        for (let d = 0; d < dims; d++) {
          dY[i][d] += (Y[i][d] - Y[j][d]) * grad_coeff;
        }
      }
    }

    // Update embeddings
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < dims; d++) {
        // Update gains
        const sign_changed = Math.sign(dY[i][d]) !== Math.sign(Y_step[i][d]);
        gains[i][d] = sign_changed ? Math.max(gains[i][d] * 0.8, 0.01) : gains[i][d] + 0.2;
        
        // Update step
        Y_step[i][d] = momentum * Y_step[i][d] - learningRate * gains[i][d] * dY[i][d];
        
        // Update position
        Y[i][d] += Y_step[i][d];
      }
    }

    // Center the embeddings
    const meanY = Array(dims).fill(0);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < dims; d++) {
        meanY[d] += Y[i][d];
      }
    }
    for (let d = 0; d < dims; d++) meanY[d] /= n;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < dims; d++) {
        Y[i][d] -= meanY[d];
      }
    }

    if ((iter + 1) % 50 === 0) {
      console.log(`Custom t-SNE Iteration ${iter + 1}/${iterations}`);
      // Optional: Calculate cost (KL divergence) here if needed
    }
  }

  console.log('Custom t-SNE finished.');
  // Return the final 2D coordinates along with original data
  return data.map((item, index) => ({
    ...item,
    value: Y[index] // The new 2D coordinates
  }));
};

// Function to process features and select representatives for charting
const findRepresentativeFeatures = () => {
  console.log('Finding representative features...');
  // Combine loaded features and add type for coloring
  // Store original 3D vector and prepare for t-SNE
  const allFeatures = [
    ...positiveFeatures.value.map(f => {
      const vector = f.feature_vector || [0, 0, 0]; // Handle potential missing vector, assume 3D
      return {
        id: f.id,
        originalValue: vector.map(Number), // Store original 3D vector
        image: f.image,
        type: 'positive'
      };
    }),
    ...negativeFeatures.value.map(f => {
      const vector = f.feature_vector || [0, 0, 0]; // Handle potential missing vector, assume 3D
      return {
        id: f.id,
        originalValue: vector.map(Number), // Store original 3D vector
        image: f.image,
        type: 'negative'
      };
    })
  ];

  // Apply custom t-SNE to reduce dimensions from 3D to 2D
  if (allFeatures.length > 0) {
    // Call the custom basicTSNE function
    representativeFeatures.value = basicTSNE(allFeatures, {
      perplexity: 30, // Adjust as needed
      learningRate: 100, // Adjust as needed
      iterations: 500, // Adjust as needed
      dims: 2
    });
  } else {
    representativeFeatures.value = [];
  }

  console.log(`Processed ${representativeFeatures.value.length} features with t-SNE.`);
  // Add a log to check the actual data being prepared
  console.log('Representative Features for Chart (Post t-SNE):', JSON.stringify(representativeFeatures.value.map(f => ({id: f.id, value: f.value})), null, 2));
};

// Function to load data from JSON files
const loadData = async () => {
  console.log('Generating random features...');
  try {
    // Generate 50 positive and 50 negative features with clustered distribution
    positiveFeatures.value = generateRandomFeatures(50, 'positive');
    negativeFeatures.value = generateRandomFeatures(50, 'negative');
    console.log('Random features generated successfully.');

    // Proceed with feature comparison and chart update
    findRepresentativeFeatures();
    updateChart();
  } catch (error) {
    console.error('Error generating data:', error);
    positiveFeatures.value = [];
    negativeFeatures.value = [];
    representativeFeatures.value = [];
    updateChart(); // Update chart to show empty state or error
  }
};

// Function to update the ECharts instance
const updateChart = () => {
  if (!chartInstance.value) {
    console.error('Chart instance not available for update.');
    return;
  }
  console.log(`Updating chart with display mode: ${displayMode.value}...`);

  // Prepare data for ECharts, including the feature ID and dynamic opacity
  const chartData = representativeFeatures.value.map(f => {
    let opacity = 0.8; // Default opacity
    if (displayMode.value === 'positive' && f.type === 'negative') {
      opacity = 0.1; // Dim negative points
    } else if (displayMode.value === 'negative' && f.type === 'positive') {
      opacity = 0.1; // Dim positive points
    }

    return {
      name: f.id, // Use feature ID for tooltip and click identification
      value: f.value, // The coordinates [dim1, dim2]
      itemStyle: {
        color: f.type === 'positive' ? '#c23531' : '#2f4554', // Different colors for groups
        opacity: opacity // Apply dynamic opacity
      },
      emphasis: { // Style for highlighted items
          itemStyle: {
              opacity: 1, // Full opacity on hover regardless of mode
              borderColor: '#000',
              borderWidth: 1.5
          }
      },
      // Store image path directly in data item if needed, though click handler uses ID
      imagePath: f.image
    };
  });
  // Add a log to check the data being passed to ECharts
  console.log('Chart Data for ECharts:', JSON.stringify(chartData.map(d => ({name: d.name, opacity: d.itemStyle.opacity})), null, 2));

  const option = {
    title: { text: '特征聚类图' }, // Updated title
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        // Find the original feature data to display more info if needed
        const feature = representativeFeatures.value.find(f => f.id === params.name);
        let tooltipText = `特征 ID: ${params.name}`;
        if (feature) {
          tooltipText += `<br/>类型: ${feature.type === 'positive' ? '阳性' : '阴性'}`;
          // Add original vector if helpful, format nicely
          // tooltipText += `<br/>原始向量: [${feature.originalValue.map(v => v.toFixed(2)).join(', ')}]`;
          tooltipText += `<br/>坐标: [${params.value[0].toFixed(2)}, ${params.value[1].toFixed(2)}]`;
        }
        return tooltipText;
      }
    },
    xAxis: { type: 'value', scale: true, name: 't-SNE 维度1' }, // Auto scale axis
    yAxis: { type: 'value', scale: true, name: 't-SNE 维度2' },
    series: [
      {
        name: 'Features',
        type: 'scatter',
        data: chartData,
        symbolSize: 10,
        // Enable emphasis effect on hover
        emphasis: {
          focus: 'series', // Highlight the item and optionally dim others
          scale: 1.2 // Slightly enlarge the hovered item
        }
      },
    ],
  };
  chartInstance.value.setOption(option, true); // Use true to clear previous options
  console.log('Chart updated.');
};

// Placeholder for handling chart interaction
const handleChartClick = (params) => {
  const featureId = params.name; // Get feature ID from clicked point
  if (!featureId) return;

  console.log('Chart clicked, feature ID:', featureId);
  // Find the feature in representativeFeatures based on ID
  const clickedFeature = representativeFeatures.value.find(f => f.id === featureId);

  if (clickedFeature && clickedFeature.image) {
    selectedImage.value = clickedFeature.image; // Update selected image
    console.log('Selected image updated:', selectedImage.value);

    // Find nearby images (simple example: find N closest points in the 2D space)
    findNearbyImages(clickedFeature.value, 5); // Find 5 nearest neighbors
    
    // Get nearby points coordinates for outline
    const distances = representativeFeatures.value
      .map((feature, index) => {
        if (!feature.value) return null;
        const dx = feature.value[0] - clickedFeature.value[0];
        const dy = feature.value[1] - clickedFeature.value[1];
        const distSq = dx * dx + dy * dy;
        return { index, distSq, coords: feature.value };
      })
      .filter(item => item !== null && item.distSq < 0.1); // Adjust threshold as needed
    
    // Sort by distance and take top 10
    distances.sort((a, b) => a.distSq - b.distSq);
    const nearbyPoints = distances.slice(0, 10).map(item => item.coords);
    
    // Calculate convex hull for outline
    const hullPoints = calculateConvexHull(nearbyPoints);
    console.log('Convex hull points:', hullPoints);
    
    // Ensure hullPoints has at least 3 points to form a polygon
    if (hullPoints.length >= 3) {
      // Update chart with outline
      const option = chartInstance.value.getOption();
      option.graphic = option.graphic || [];
      // Clear existing graphics first
      option.graphic = option.graphic.filter(g => g.type !== 'polygon');
      option.graphic.push({
        type: 'polygon',
        shape: {
          points: hullPoints
        },
        style: {
          fill: 'rgba(255, 0, 0, 0.1)',
          stroke: '#ff0000',
          lineWidth: 3,
          shadowBlur: 10,
          shadowColor: '#ff0000'
        },
        zlevel: 10
      });
      
      // Add outline for the clicked point
      option.graphic.push({
        type: 'circle',
        shape: {
          cx: clickedFeature.value[0],
          cy: clickedFeature.value[1],
          r: 5
        },
        style: {
          fill: '#ff0000',
          stroke: '#ffffff',
          lineWidth: 2
        },
        zlevel: 11
      });
      
      chartInstance.value.setOption(option, true);
    } else {
      console.log('Not enough points to form a convex hull');
    }

  } else {
    console.log('Clicked feature or image path not found for ID:', featureId);
    selectedImage.value = null; // Clear selection if not found
    nearbyImages.value = []; // Clear nearby images
    
    // Clear outline
    const option = chartInstance.value.getOption();
    option.graphic = [];
    chartInstance.value.setOption(option);
  }
};

// Convex hull algorithm (Graham scan)
const calculateConvexHull = (points) => {
  if (points.length < 3) return points;
  
  // Find point with lowest y (and leftmost if tie)
  let pivot = points[0];
  for (const p of points) {
    if (p[1] < pivot[1] || (p[1] === pivot[1] && p[0] < pivot[0])) {
      pivot = p;
    }
  }
  
  // Sort by polar angle from pivot
  const sorted = [...points].sort((a, b) => {
    const angleA = Math.atan2(a[1] - pivot[1], a[0] - pivot[0]);
    const angleB = Math.atan2(b[1] - pivot[1], b[0] - pivot[0]);
    return angleA - angleB;
  });
  
  // Build hull
  const hull = [sorted[0], sorted[1]];
  for (let i = 2; i < sorted.length; i++) {
    while (hull.length >= 2 && crossProduct(hull[hull.length-2], hull[hull.length-1], sorted[i]) <= 0) {
      hull.pop();
    }
    hull.push(sorted[i]);
  }
  
  return hull;
};

// Helper for convex hull
const crossProduct = (a, b, c) => {
  return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]);
};

// Function to find nearby images based on 2D coordinates
const findNearbyImages = (targetCoords, count) => {
  if (!representativeFeatures.value || representativeFeatures.value.length === 0) {
    nearbyImages.value = [];
    return;
  }

  // Calculate distances from the target point to all other points
  const distances = representativeFeatures.value
    .map((feature, index) => {
      if (!feature.value) return null; // Skip if no coordinates
      const dx = feature.value[0] - targetCoords[0];
      const dy = feature.value[1] - targetCoords[1];
      const distSq = dx * dx + dy * dy;
      return { index, distSq, image: feature.image, id: feature.id };
    })
    .filter(item => item !== null && item.id !== representativeFeatures.value.find(f => f.value === targetCoords)?.id); // Exclude the point itself

  // Sort by distance (ascending)
  distances.sort((a, b) => a.distSq - b.distSq);

  // Get the top 'count' nearest neighbors
  nearbyImages.value = distances.slice(0, count).map(item => item.image);
  console.log('Nearby images found:', nearbyImages.value);
};

// --- ECharts Event Handlers ---
const handleChartHover = (params) => {
  // ECharts handles hover emphasis via the 'emphasis' configuration in the series.
  // No manual dispatchAction needed for basic hover highlight.
  // console.log('Hovering over:', params.name);
  // Logic to update selectedImage and nearbyImages on hover is removed.
  // Hover effect is purely visual via ECharts emphasis config.
};

const handleChartMouseOut = (params) => {
  // ECharts handles resetting emphasis automatically.
  // console.log('Mouse out from:', params.name);
  // Logic to clear selectedImage and nearbyImages on mouse out is removed.
  // Selection persists until a new point is clicked.
};

// --- Vue Lifecycle Hooks ---
onMounted(() => {
  const chartDom = document.getElementById('echarts-placeholder');
  if (chartDom) {
    chartInstance.value = echarts.init(chartDom);
    // Set initial empty options or loading state
    chartInstance.value.setOption({
      title: { text: '加载中...' },
      tooltip: {},
      xAxis: { type: 'value' },
      yAxis: { type: 'value' },
      series: []
    });
    // Add click listener
    chartInstance.value.on('click', handleChartClick);
    // Add hover listener
    chartInstance.value.on('mouseover', handleChartHover);
    // Add mouseout listener to reset highlights
    chartInstance.value.on('mouseout', handleChartMouseOut);
    // Load data and update chart
    loadData(); // This will now populate and update the chart
  } else {
    console.error('ECharts container not found');
  }

  // Optional: Resize chart with window resize
  const resizeHandler = () => {
    chartInstance.value?.resize();
  };
  window.addEventListener('resize', resizeHandler);

  // Cleanup on unmount
  onUnmounted(() => {
    window.removeEventListener('resize', resizeHandler);
    chartInstance.value?.dispose();
  });
});

// Cleanup on unmount
// onUnmounted is already imported and used within onMounted

// Watch for changes in displayMode and update the chart
watch(displayMode, () => {
  updateChart();
});

</script>

<template>
  <div id="app-container">
    <h1>病理图像特征分析与可视化</h1>
    <div class="main-content">
      <div class="chart-container">
        <div class="chart-header">
          <h2>特征聚类图</h2>
          <!-- Display Mode Controls -->
          <div class="display-controls">
            <label>
              <input type="radio" v-model="displayMode" value="all"> 全部
            </label>
            <label>
              <input type="radio" v-model="displayMode" value="positive"> 仅阳性
            </label>
            <label>
              <input type="radio" v-model="displayMode" value="negative"> 仅阴性
            </label>
          </div>
        </div>
        <div id="echarts-placeholder"></div> <!-- Removed inline style -->
      </div>
      <!-- Combined Image Display Area -->
      <div class="image-display-area">
        <!-- Selected Image Display -->
        <div class="selected-image-container">
          <h3>选定图像</h3>
          <div v-if="selectedImage">
            <img :src="selectedImage" alt="Selected Feature Image">
            <!-- Optional: Display path for debugging -->
            <!-- <p style="font-size: 0.8em; word-break: break-all;">路径: {{ selectedImage }}</p> -->
          </div>
          <div v-else>
            <p>点击图中的点以显示图像</p>
          </div>
        </div>
        <!-- Nearby Images Display -->
        <div class="nearby-images-container">
           <h3>选定点附近图像</h3>
          <div v-if="nearbyImages.length > 0" class="nearby-images">
            <!-- Ensure image paths are correct relative to public folder -->
            <img v-for="(img, index) in nearbyImages.filter(i => i)" :key="index" :src="img" alt="Nearby Feature Image">
          </div>
           <div v-else>
            <p>点击图中的点以显示附近图像</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
#app-container {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  padding: 30px; /* Increased padding */
  /* max-width: 1500px; /* Ensure no max-width */
  margin: 30px 0; /* 30px top/bottom, 0 left/right for full width */
  background-color: #f8f9fa; /* Light background for the whole app */
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Subtle shadow */
}

h1 {
  text-align: center;
  margin-bottom: 40px; /* Increased spacing */
  color: #343a40; /* Darker heading color */
  font-weight: 600;
}

.main-content {
  display: flex;
  flex-direction: row; /* Arrange chart and image display side-by-side */
  gap: 40px; /* Increased space between main sections */
  margin-top: 20px;
}

.chart-container {
  flex: 3; /* Give chart more space (e.g., ~60%) */
  min-width: 500px; /* Ensure chart has a reasonable minimum width */
  display: flex;
  flex-direction: column;
  background-color: #ffffff; /* White background for chart area */
  padding: 20px;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px; /* Space below header */
}

.chart-header h2 {
  margin: 0; /* Remove default margin */
  text-align: left;
}

.display-controls {
  display: flex;
  gap: 15px; /* Space between radio buttons */
}

.display-controls label {
  cursor: pointer;
  font-size: 0.9em;
  color: #495057;
}

.display-controls input[type="radio"] {
  margin-right: 5px;
}

#echarts-placeholder {
  width: 100%;
  height: 600px; /* Increased chart height */
  border: 1px solid #dee2e6; /* Lighter border */
  border-radius: 4px;
}

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
    min-height: 320px; /* Ensure minimum height */
    justify-content: center; /* Center content vertically */
}

.nearby-images-container {
    flex-grow: 1; /* Allow nearby container to grow */
}

h2, h3 {
  text-align: center;
  margin-top: 0;
  margin-bottom: 20px; /* Increased spacing */
  color: #495057; /* Slightly lighter heading color */
  font-weight: 500;
}

.selected-image-container img {
  display: block;
  max-width: 100%;
  max-height: 350px; /* Adjust max height for selected image */
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
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); /* Slightly larger grid items */
  gap: 15px;
  max-height: 400px; /* Adjust height */
  overflow-y: auto; /* Keep scroll for overflow */
  margin-top: 10px;
  padding-right: 5px; /* Add padding for scrollbar */
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

/* Responsive adjustments */
@media (max-width: 1000px) { /* Adjusted breakpoint */
  .main-content {
    flex-direction: column;
  }
  .chart-container,
  .image-display-area {
    flex: 1; /* Reset flex basis */
    min-width: 100%; /* Take full width */
  }
  #echarts-placeholder {
    height: 500px; /* Adjust height for medium screens */
  }
  .chart-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  .display-controls {
    align-self: flex-start; /* Align controls to the left */
  }
}

@media (max-width: 600px) { /* Added breakpoint for smaller screens */
  #app-container {
    padding: 15px;
    margin: 15px auto;
  }
  h1 {
    font-size: 1.5em;
    margin-bottom: 25px;
  }
  .main-content {
    gap: 20px;
  }
  #echarts-placeholder {
    height: 400px; /* Further adjust height */
  }
  .selected-image-container,
  .nearby-images-container {
    padding: 15px;
  }
  h2, h3 {
    font-size: 1.1em;
    margin-bottom: 15px;
  }
  .nearby-images {
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); /* Smaller grid items */
    gap: 10px;
  }
  .nearby-images img {
    height: 80px;
  }
  .chart-header {
    align-items: center;
  }
  .display-controls {
    flex-wrap: wrap; /* Allow controls to wrap */
    justify-content: center;
    gap: 10px;
    align-self: center;
  }
}
</style>
