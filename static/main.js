// Global loading overlay control
function showGlobalLoading() {
  const el = document.getElementById("global-loading");
  if (el) el.classList.remove("hidden");
}
function hideGlobalLoading() {
  const el = document.getElementById("global-loading");
  if (el) el.classList.add("hidden");
}

window.addEventListener("load", () => {
  const runForm = document.getElementById("run-probes-form");
  if (runForm) {
    runForm.addEventListener("submit", () => {
      showGlobalLoading();
    });
  }

  const compareForm = document.getElementById("compare-form");
  if (compareForm) {
    compareForm.addEventListener("submit", () => {
      showGlobalLoading();
    });
  }

  const datasetForm = document.querySelector(".dataset-form");
  if (datasetForm) {
    datasetForm.addEventListener("submit", () => {
      showGlobalLoading();
    });
  }
});
