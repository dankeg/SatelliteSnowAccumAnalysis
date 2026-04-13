const fileInput = document.getElementById("imageFile");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const inputPreviewEl = document.getElementById("inputPreview");
const maskPreviewEl = document.getElementById("maskPreview");
const overlayPreviewEl = document.getElementById("overlayPreview");

let selectedFile = null;

function setPreviewImage(element, file) {
  if (!file || !file.type.startsWith("image/")) {
    element.removeAttribute("src");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    element.src = reader.result;
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files?.[0] ?? null;
  predictBtn.disabled = !selectedFile;
  statusEl.textContent = selectedFile
    ? `Ready: ${selectedFile.name}`
    : "No file selected.";

  setPreviewImage(inputPreviewEl, selectedFile);
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  predictBtn.disabled = true;
  statusEl.textContent = "Running model...";
  outputEl.textContent = "Submitting file to the backend...";
  maskPreviewEl.removeAttribute("src");
  overlayPreviewEl.removeAttribute("src");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    maskPreviewEl.src = `data:image/png;base64,${data.prediction_mask_base64}`;
    overlayPreviewEl.src = `data:image/png;base64,${data.overlay_base64}`;
    outputEl.textContent = JSON.stringify(data, null, 2);
    statusEl.textContent = `Done. Snow fraction: ${(data.snow_fraction * 100).toFixed(1)}%`;
  } catch (error) {
    outputEl.textContent = `Error: ${error.message || error}`;
    statusEl.textContent = "Prediction failed.";
  } finally {
    predictBtn.disabled = !selectedFile;
  }
});

