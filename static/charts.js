// Color palette
const palette = [
  "#22d3ee",
  "#a855f7",
  "#f97316",
  "#22c55e",
  "#eab308",
  "#fb7185",
];

// ----------------- RUN RESULTS (single run) -----------------

function buildLayerDatasets(metrics, metricKey) {
  const modelNames = Object.keys(metrics);
  const labels = [];

  const firstModel = metrics[modelNames[0]];
  firstModel.forEach((m) => labels.push(m.layer_index));

  const datasets = modelNames.map((m, idx) => {
    const data = metrics[m].map((lm) => lm[metricKey]);
    return {
      label: m,
      data,
      borderColor: palette[idx % palette.length],
      backgroundColor: palette[idx % palette.length] + "66",
      tension: 0.25,
    };
  });

  return { labels, datasets };
}

function renderRunCharts(metrics) {
  const { labels, datasets } = buildLayerDatasets(metrics, "accuracy");
  const f1 = buildLayerDatasets(metrics, "f1");
  const ece = buildLayerDatasets(metrics, "ece");

  const ctxAcc = document.getElementById("accChart");
  const ctxF1 = document.getElementById("f1Chart");
  const ctxECE = document.getElementById("eceChart");

  if (ctxAcc) {
    new Chart(ctxAcc, {
      type: "line",
      data: { labels, datasets },
      options: baseLineOptions("Layer Index", "Accuracy", 0, 1),
    });
  }

  if (ctxF1) {
    new Chart(ctxF1, {
      type: "line",
      data: { labels: f1.labels, datasets: f1.datasets },
      options: baseLineOptions("Layer Index", "Weighted F1", 0, 1),
    });
  }

  if (ctxECE) {
    new Chart(ctxECE, {
      type: "line",
      data: { labels: ece.labels, datasets: ece.datasets },
      options: baseLineOptions("Layer Index", "ECE", 0, 0.5),
    });
  }

  setTimeout(() => {
    if (typeof hideGlobalLoading === "function") hideGlobalLoading();
  }, 300);
}

function baseLineOptions(xLabel, yLabel, yMin, yMax) {
  return {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#e5e5e5" } },
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, color: "#9ca3af" },
        ticks: { color: "#9ca3af" },
      },
      y: {
        title: { display: true, text: yLabel, color: "#9ca3af" },
        min: yMin,
        max: yMax,
        ticks: { color: "#9ca3af" },
      },
    },
  };
}

// ----------------- INLINE COMPARE (instant) -----------------

let compareChart = null;

function renderInlineCompareCharts(metricsA, metricsB, modelA, modelB) {
  const labels = metricsA.map((m) => m.layer_index);

  function getMetricData(key) {
    return {
      labels,
      datasets: [
        {
          label: modelA,
          data: metricsA.map((m) => m[key]),
          borderColor: palette[0],
          backgroundColor: palette[0] + "66",
          tension: 0.25,
        },
        {
          label: modelB,
          data: metricsB.map((m) => m[key]),
          borderColor: palette[1],
          backgroundColor: palette[1] + "66",
          tension: 0.25,
        },
      ],
    };
  }

  const ctx = document.getElementById("compareMetricChart");
  if (!ctx) return;

  const defaultMetric = "acc";
  const metricConfig = {
    acc: { key: "accuracy", label: "Accuracy", min: 0, max: 1 },
    f1: { key: "f1", label: "Weighted F1", min: 0, max: 1 },
    ece: { key: "ece", label: "ECE", min: 0, max: 0.5 },
  };

  function draw(metricKey) {
    const cfg = metricConfig[metricKey];
    const data = getMetricData(cfg.key);

    if (compareChart) {
      compareChart.destroy();
    }

    compareChart = new Chart(ctx, {
      type: "line",
      data,
      options: baseLineOptions("Layer Index", cfg.label, cfg.min, cfg.max),
    });
  }

  // Initial render
  draw(defaultMetric);

  // Wire up chips
  const chips = document.querySelectorAll(".metric-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("metric-chip-active"));
      chip.classList.add("metric-chip-active");
      const metric = chip.getAttribute("data-metric");
      draw(metric);
    });
  });

  setTimeout(() => {
    if (typeof hideGlobalLoading === "function") hideGlobalLoading();
  }, 300);
}

function renderInsightsCharts(data) {
  if (!data) return;

  const metric = data.metric;
  const models = data.models || [];
  const layers = data.layers || [];

  const metricLabel =
    metric === "accuracy"
      ? "Accuracy"
      : metric === "f1"
      ? "Weighted F1"
      : "ECE (lower is better)";

  // Model/dataset bar chart
  const mCtx = document.getElementById("insightsModelChart");
  if (mCtx && models.length) {
    const labels = models.map((m) => `${m.model}\n${m.dataset}`);
    const values = models.map((m) => m.best);

    new Chart(mCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: `Best ${metricLabel}`,
            data: values,
            backgroundColor: "#22d3ee66",
            borderColor: "#22d3ee",
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            ticks: {
              color: "#9ca3af",
              font: { size: 10 },
            },
          },
          y: {
            ticks: {
              color: "#9ca3af",
              font: { size: 10 },
            },
          },
        },
      },
    });
  }

  // Layer signature line chart
  const lCtx = document.getElementById("insightsLayerChart");
  if (lCtx && layers.length) {
    const labels = layers.map((l) => l.layer);
    const values = layers.map((l) => l.mean);

    new Chart(lCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: `Mean ${metricLabel}`,
            data: values,
            borderColor: "#a855f7",
            backgroundColor: "#a855f766",
            tension: 0.25,
          },
        ],
      },
      options: baseLineOptions("Layer index", metricLabel, null, null),
    });
  }
}

window.renderInsightsCharts = renderInsightsCharts;
