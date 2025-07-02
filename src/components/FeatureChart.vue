<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import 'echarts-gl';

const props = defineProps({
  representativeFeatures: Array,
  displayMode: String
});

const emit = defineEmits(['chart-click']);

const chartInstance = ref(null);

// Helper function for Euclidean distance squared (local to this component or import if widely used)
const euclideanDistSq = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return sum;
};

// 2D Convex hull algorithm (Graham scan) - Not directly applicable for 3D scatter plot interaction in this context
// const calculateConvexHull = (points) => { ... };
// const crossProduct = (a, b, c) => { ... };

// Function to update the ECharts instance
const updateChart = () => {
  if (!chartInstance.value || !props.representativeFeatures) {
    console.error('Chart instance or representativeFeatures not available for update.');
    return;
  }
  console.log(`Updating chart with display mode: ${props.displayMode}...`);

  // 1. 应用显示模式过滤
  let filteredFeatures = [...props.representativeFeatures];
  if (props.displayMode === 'positiveonly') {
    filteredFeatures = filteredFeatures.filter(f => f.type === 1);
  } else if (props.displayMode === 'negativeonly') {
    filteredFeatures = filteredFeatures.filter(f => f.type === 0);
  }

  // 2. 处理数据和样式
  const chartData = filteredFeatures.map(f => ({
    name: f.id,
    value: getPointCoordinates(f),
    itemStyle: getPointStyle(f),
    emphasis: getEmphasisStyle(f),
    symbolSize: getPointSize(f),
    zlevel: f.is_domain ? 2 : 1, // domain点置于顶层，避免被遮挡
    imagePath: f.image, // Keep imagePath for potential use in tooltips or clicks
    // Store original 2D/3D data if needed for specific interactions
    originalUMAP2d: f.value, // This is the original UMAP_2d from loadData
    originalUMAP3d: f.UMAP_3d // This is the original UMAP_3d from loadData
  }));

  const option = {
    title: { text: '特征聚类图 (3D)' },
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        // params.data should be the feature object from chartData
        const featureData = params.data;
        let tooltipText = `特征 ID: ${featureData.name}`;
        // Access original feature type from props.representativeFeatures if needed, or ensure it's in chartData
        const originalFeature = props.representativeFeatures.find(f => f.id === featureData.name);

        if (originalFeature) {
          tooltipText += `<br/>类型: ${originalFeature.type === 1 ? '阳性' : '阴性'}`;
        }

        if (featureData.value && featureData.value.length === 3) {
          tooltipText += `<br/>坐标 (3D): [${featureData.value[0].toFixed(2)}, ${featureData.value[1].toFixed(2)}, ${featureData.value[2].toFixed(2)}]`;
        } else if (originalFeature && originalFeature.value && originalFeature.value.length === 2) { // Fallback to original 2D if 3D rendering used Z=0
           tooltipText += `<br/>坐标 (2D): [${originalFeature.value[0].toFixed(2)}, ${originalFeature.value[1].toFixed(2)}]`;
        }
        return tooltipText;
      }
    },
    grid3D: {
      viewControl: {
        // autoRotate: true, // Optional: for automatic rotation
        distance: 150 // Adjust camera distance
      }
    },
    xAxis3D: { type: 'value', name: 'UMAP Dim 1' },
    yAxis3D: { type: 'value', name: 'UMAP Dim 2' },
    zAxis3D: { type: 'value', name: 'UMAP Dim 3' },
    series: [
      {
        name: 'Features',
        type: 'scatter3D',
        data: chartData, // chartData now contains objects with 'value' as 3D array
        symbolSize: 8, // Adjust symbol size for 3D view
        emphasis: {
          focus: 'series',
          label: {
            show: false // Don't show label on emphasis to keep it clean
          },
          itemStyle: {
            opacity: 1,
            borderColor: '#000',
            borderWidth: 1.5
          }
        }
      },
    ]
  };
  // Clear previous 2D graphics if any when switching to 3D
  const currentOption = chartInstance.value.getOption();
  if (currentOption && currentOption.graphic) {
    option.graphic = []; // Clear 2D graphics like hull
  }
  chartInstance.value.setOption(option, true);
  console.log('Chart updated.');
};

const handleChartClickInternal = (params) => {
  const featureId = params.name;
  if (!featureId) return;

  console.log('Chart clicked in FeatureChart, feature ID:', featureId);
  const clickedFeature = props.representativeFeatures.find(f => f.id === featureId);

  if (clickedFeature) {
    emit('chart-click', clickedFeature);
    // Convex hull drawing is primarily for 2D and might not be directly applicable or visually clear in 3D.
    // clearConvexHull(); // Ensure any 2D hull is cleared if it was drawn before.
  } else {
    // Clear outline if clicked on empty space or invalid point
    clearConvexHull();
    emit('chart-click', null); // Emit null if feature not found
  }
};

// const drawConvexHullAroundPoint = (clickedFeature) => { ... }; // 2D specific, removed for 3D
// const clearConvexHull = () => { ... }; // 2D specific, removed for 3D


onMounted(() => {
  const chartDom = document.getElementById('echarts-scatter-plot');
  if (chartDom) {
    chartInstance.value = echarts.init(chartDom);
    chartInstance.value.setOption({
      title: { text: '加载中...' },
      tooltip: {}, xAxis: { type: 'value' }, yAxis: { type: 'value' }, series: []
    });
    chartInstance.value.on('click', handleChartClickInternal);
    // Initial chart update if data is already available
    if (props.representativeFeatures && props.representativeFeatures.length > 0) {
        updateChart();
    }
  } else {
    console.error('ECharts container (echarts-scatter-plot) not found');
  }

  const resizeHandler = () => chartInstance.value?.resize();
  window.addEventListener('resize', resizeHandler);
  onUnmounted(() => {
    window.removeEventListener('resize', resizeHandler);
    chartInstance.value?.dispose();
  });
});
// 抽取坐标处理逻辑
const getPointCoordinates = (feature) => {
  if (feature.UMAP_3d && Array.isArray(feature.UMAP_3d) && feature.UMAP_3d.length === 3) {
    return feature.UMAP_3d;
  } else if (feature.value && Array.isArray(feature.value) && feature.value.length === 2) {
    return [feature.value[0], feature.value[1], 0]; // 2D坐标转3D
  }
  console.warn(`Feature ${feature.id} has no valid coordinates`);
  return [0, 0, 0];
};

// 抽取样式处理逻辑 - 核心重构部分
const getPointStyle = (feature) => {
  // 基础样式
  const baseStyle = {
    color: getBaseColor(feature),
    opacity: feature.is_domain ? 1.0 : 0.2, // 降低非domain点存在感
  };

  // domain点高亮效果
  if (feature.is_domain) {
    return {
      ...baseStyle,
      shadowBlur: 10 + (feature.ratio || 0) * 15, // 根据ratio调整发光强度
      shadowColor: baseStyle.color,
      shadowOffsetX: 0,
      shadowOffsetY: 0,
      borderColor: '#fff',
      borderWidth: 1.5
    };
  }

  return baseStyle;
};

// 抽取颜色逻辑
const getBaseColor = (feature) => {
  // 保持原有阳性/阴性颜色区分
  return feature.type === 1 ? '#c23531' : '#2f4554';
};

// 抽取大小逻辑
const getPointSize = (feature) => {
  if (feature.is_domain) {
    // domain点更大且根据ratio调整
    return 10 + (feature.ratio || 0) * 8;
  }
  // 非domain点更小
  return 6;
};

// 抽取强调样式逻辑
const getEmphasisStyle = (feature) => ({
  itemStyle: {
    opacity: 1,
    borderColor: '#000',
    borderWidth: 2
  }
});
watch(() => props.representativeFeatures, (newVal) => {
  if (newVal) updateChart();
}, { deep: true });

watch(() => props.displayMode, () => {
  updateChart();
});

// Expose methods to parent if needed, though props/events are preferred
// defineExpose({ updateChart });

</script>

<template>
  <div id="echarts-scatter-plot" class="chart-canvas"></div>
</template>

<style scoped>
.chart-canvas {
  width: 100%;
  height: 600px !important; /* Or make this configurable via props */
  border: 1px solid #dee2e6;
  border-radius: 4px;
}
</style>


