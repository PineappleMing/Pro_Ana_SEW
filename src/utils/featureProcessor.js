// src/utils/featureProcessor.js

// Helper function for Euclidean distance squared
export const euclideanDistSq = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return sum;
};

// Custom t-SNE implementation
export const basicTSNE = (data, { perplexity = 30, learningRate = 100, iterations = 500, dims = 2 }) => {
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
    }
  }

  console.log('Custom t-SNE finished.');
  return data.map((item, index) => ({
    ...item,
    value: Y[index]
  }));
};

// Random data generator with clustered distribution
export const generateRandomFeatures = (count, type) => {
  const centers = type === 'positive' ?
    [
      [0.3, 0.3, 0.1],
      [0.5, 0.1, 0.3],
      [0.1, 0.5, 0.2],
      [0.4, -0.2, 0.2],
      [-0.2, 0.4, 0.0]
    ] :
    [
      [-0.3, -0.3, -0.1],
      [-0.5, -0.1, -0.3],
      [-0.1, -0.5, -0.2],
      [-0.4, 0.2, -0.2],
      [0.2, -0.4, 0.0]
    ];

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
        center[0] + (Math.random() * 1.0 - 0.5),
        center[1] + (Math.random() * 1.0 - 0.5),
        center[2] + (Math.random() * 1.0 - 0.5)
      ]
    };
    features.push(feature);
  }
  
  return features;
};

// The findRepresentativeFeatures function might not be strictly necessary
// if all loaded data points with UMAP_2d coordinates are to be displayed.
// If a specific selection/sampling is still needed, it can be adapted or kept.
// For now, we assume all processed features with UMAP_2d are representative.
/*
const findRepresentativeFeatures = (features, count) => {
  if (features.length <= count) {
    return features;
  }
  const shuffled = [...features].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
};
*/

// Function to load and process data from JSON files
export const loadData = async () => {
  try {
    // Fetch data from the new JSON file structure
    // Assuming your new data is in a single file, adjust if it's multiple files
    const response = await fetch('/clustered_patch_data.json'); // Or the correct path to your new data file
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const rawData = await response.json();

    // Adapt the rawData to the structure expected by the application
    const processedFeatures = rawData.map(item => ({
      id: item.slide_id + '_' + item.patch_coord.join('_'), // Create a unique ID
      image: item.image_url,
      type: item.label, // 'positive' or 'negative'
      value: item.UMAP_2d, // Directly use the 2D UMAP coordinates
      // Add other necessary fields if any, e.g., originalValue if still needed for some processing
      // originalValue: item.embedding, // If you still need the original embedding for some reason
      patch_coord: item.patch_coord,
      slide_id: item.slide_id,
      UMAP_3d: item.UMAP_3d // Keep 3D coordinates if needed for other purposes
    }));

    // Separate features based on label
    const positiveFeatures = processedFeatures.filter(f => f.type === "1");
    const negativeFeatures = processedFeatures.filter(f => f.type === "0");
    // Since we are directly using UMAP_2d, all loaded features can be considered 'representative'
    // Or, if you still need a subset, you can implement a selection logic here.
    // For now, let's assume all processed features are used directly.
    const representativeFeatures = processedFeatures;

    return {
      positiveFeatures,
      negativeFeatures,
      representativeFeatures // These now directly use UMAP_2d coordinates
    };
  } catch (error) {
    console.error("Error loading or processing data:", error);
    return { positiveFeatures: [], negativeFeatures: [], representativeFeatures: [] };
  }
};