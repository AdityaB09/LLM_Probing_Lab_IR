function setupTokenHeatmap(attribution) {
  if (!attribution) return;

  const container = document.getElementById("token-heatmap");
  const slider = document.getElementById("layer-slider");
  const layerLabel = document.getElementById("layer-label");

  const tokens = attribution.tokens;
  const layers = attribution.layers;

  if (!container || !slider || !layers.length) return;

  // Configure slider range
  const minLayer = layers[0].layer_index;
  const maxLayer = layers[layers.length - 1].layer_index;
  slider.min = minLayer;
  slider.max = maxLayer;
  slider.value = minLayer;

  function findLayer(idx) {
    return (
      layers.find((l) => l.layer_index === idx) ||
      layers[0]
    );
  }

  function render(idx) {
    const layer = findLayer(idx);
    const scores = layer.scores;
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const denom = max - min || 1;

    layerLabel.textContent = `Layer ${layer.layer_index}`;

    container.innerHTML = "";
    tokens.forEach((tok, i) => {
      const raw = scores[i];
      const norm = (raw - min) / denom; // 0..1
      const alpha = 0.1 + 0.9 * norm;

      const span = document.createElement("span");
      span.className = "token-chip";
      span.textContent = tok.replace(/^##/, "");
      span.style.backgroundColor = `rgba(34,211,238,${alpha})`;
      span.title = `Score: ${raw.toFixed(4)}`;
      container.appendChild(span);
      container.append(" ");
    });
  }

  slider.addEventListener("input", () => {
    const idx = parseInt(slider.value, 10);
    render(idx);
  });

  // Initial render
  render(minLayer);
}

window.setupTokenHeatmap = setupTokenHeatmap;
