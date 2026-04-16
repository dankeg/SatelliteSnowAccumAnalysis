const citySelect = document.getElementById("citySelect");
const sampleGrid = document.getElementById("sampleGrid");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

const inputPreviewEl = document.getElementById("inputPreview");
const inputPlaceholderEl = document.getElementById("inputPlaceholder");
const maskPreviewEl = document.getElementById("maskPreview");
const maskPlaceholderEl = document.getElementById("maskPlaceholder");
const overlayPreviewEl = document.getElementById("overlayPreview");
const overlayPlaceholderEl = document.getElementById("overlayPlaceholder");

let cities = [];
let selectedCity = null;
let selectedSample = null;

function showFrame(imageEl, placeholderEl, src, altText) {
  if (src) {
    imageEl.src = src;
    imageEl.alt = altText;
    imageEl.style.display = "block";
    placeholderEl.style.display = "none";
    return;
  }

  imageEl.removeAttribute("src");
  imageEl.style.display = "none";
  placeholderEl.style.display = "grid";
}

function resetResults(message = "The snow mask will appear here after a run.") {
  showFrame(maskPreviewEl, maskPlaceholderEl, "", "Predicted snow mask");
  showFrame(overlayPreviewEl, overlayPlaceholderEl, "", "Overlay of snow prediction");
  outputEl.textContent = message;
}

function findCity(cityId) {
  return cities.find((city) => city.id === cityId);
}

function setSample(city, sample) {
  selectedCity = city;
  selectedSample = sample;
  runBtn.disabled = !selectedSample;

  if (!city || !sample) {
    showFrame(inputPreviewEl, inputPlaceholderEl, "", "Selected sample preview");
    return;
  }

  showFrame(inputPreviewEl, inputPlaceholderEl, sample.image_url, sample.label);
  statusEl.textContent = `Selected ${city.label}.`;
}

function renderSamples(cityId) {
  const city = findCity(cityId);
  if (!city) {
    sampleGrid.innerHTML = "";
    setSample(null, null);
    resetResults("Waiting for a sample run...");
    return;
  }

  const selected = city.id === selectedCity?.id ? selectedSample : city.samples[0];

  sampleGrid.innerHTML = "";
  city.samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = sample.id === selected?.id ? "sample-card active" : "sample-card";
    button.innerHTML = `
      <img src="${sample.image_url}" alt="${city.label} sample ${index + 1}" />
      <span>${sample.label}</span>
    `;
    button.addEventListener("click", () => {
      setSample(city, sample);
      resetResults();
      renderSamples(city.id);
    });
    sampleGrid.appendChild(button);
  });

  setSample(city, selected);
  resetResults();
}

async function loadCatalog() {
  const response = await fetch("/demo-config");
  if (!response.ok) {
    throw new Error("Could not load demo image catalog.");
  }

  const data = await response.json();
  cities = data.cities || [];

  if (!cities.length) {
    throw new Error("No sample imagery found.");
  }

  citySelect.innerHTML = cities
    .map((city) => `<option value="${city.id}">${city.label}</option>`)
    .join("");

  citySelect.value = data.default_city || cities[0].id;
  renderSamples(citySelect.value);
}

citySelect.addEventListener("change", () => {
  renderSamples(citySelect.value);
});

runBtn.addEventListener("click", async () => {
  if (!selectedSample) {
    return;
  }

  runBtn.disabled = true;
  statusEl.textContent = "Running model...";
  outputEl.textContent = "Submitting the selected image to the backend...";
  showFrame(maskPreviewEl, maskPlaceholderEl, "", "Predicted snow mask");
  showFrame(overlayPreviewEl, overlayPlaceholderEl, "", "Overlay of snow prediction");

  try {
    const imageResponse = await fetch(selectedSample.image_url);
    if (!imageResponse.ok) {
      throw new Error("Failed to load the selected sample image.");
    }

    const blob = await imageResponse.blob();
    const formData = new FormData();
    formData.append("image", new File([blob], `${selectedSample.id}.png`, { type: blob.type || "image/png" }));

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    showFrame(maskPreviewEl, maskPlaceholderEl, `data:image/png;base64,${data.prediction_mask_base64}`, "Predicted snow mask");
    showFrame(overlayPreviewEl, overlayPlaceholderEl, `data:image/png;base64,${data.overlay_base64}`, "Overlay of snow prediction");
    outputEl.textContent = JSON.stringify(data, null, 2);
    statusEl.textContent = `Done. Snow fraction: ${(data.snow_fraction * 100).toFixed(1)}%`;
  } catch (error) {
    outputEl.textContent = `Error: ${error.message || error}`;
    statusEl.textContent = "Prediction failed.";
  } finally {
    runBtn.disabled = !selectedSample;
  }
});

(async () => {
  try {
    await loadCatalog();
    statusEl.textContent = "Choose a city and sample image.";
  } catch (error) {
    statusEl.textContent = "Demo failed to load.";
    outputEl.textContent = `Error: ${error.message || error}`;
    runBtn.disabled = true;
  }
})();
